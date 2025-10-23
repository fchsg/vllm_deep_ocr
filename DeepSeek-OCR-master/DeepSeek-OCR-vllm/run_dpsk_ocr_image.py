import asyncio
import re
import os
import argparse
import ast
import time
import unicodedata

import torch

if torch.version.cuda == '11.8':
    os.environ["TRITON_PTXAS_PATH"] = "/usr/local/cuda-11.8/bin/ptxas"

os.environ['VLLM_USE_V1'] = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

from vllm import AsyncLLMEngine, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.model_executor.models.registry import ModelRegistry
from deepseek_ocr import DeepseekOCRForCausalLM
from PIL import Image, ImageDraw, ImageFont, ImageOps
import numpy as np
from tqdm import tqdm
from process.ngram_norepeat import NoRepeatNGramLogitsProcessor
from process.image_process import DeepseekOCRProcessor
from config import MODEL_PATH, INPUT_PATH, OUTPUT_PATH, PROMPT, CROP_MODE

ModelRegistry.register_model("DeepseekOCRForCausalLM", DeepseekOCRForCausalLM)


def parse_args():
    parser = argparse.ArgumentParser(description="DeepSeek-OCR vLLM 图片处理脚本")
    parser.add_argument("-i", "--input", type=str, help="输入图片路径，覆盖 config.INPUT_PATH")
    parser.add_argument("-o", "--output", type=str, help="输出目录路径，覆盖 config.OUTPUT_PATH")
    parser.add_argument("-p", "--prompt", type=str, help="覆盖默认 PROMPT（若包含 <image> 则启用图像处理）")
    parser.add_argument("--crop-mode", type=str, help="覆盖 CROP_MODE（需与项目定义一致）")
    parser.add_argument("-s", "--save", type=int, default=1, help="是否保存结果文件与裁剪图(1是/0否)，默认1")
    parser.add_argument("--mode", type=str, choices=["text", "layout"], default="text",
                        help="识别模式：text=提取纯文本(默认)，layout=版面结构与裁剪")
    return parser.parse_args()


def load_image(image_path):
    try:
        image = Image.open(image_path)
        corrected_image = ImageOps.exif_transpose(image)
        return corrected_image
    except Exception as e:
        print(f"error: {e}")
        try:
            return Image.open(image_path)
        except:
            return None


def re_match(text):
    pattern = r'(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)'
    matches = re.findall(pattern, text, re.DOTALL)

    mathes_image = []
    mathes_other = []
    for a_match in matches:
        if '<|ref|>image<|/ref|>' in a_match[0]:
            mathes_image.append(a_match[0])
        else:
            mathes_other.append(a_match[0])
    return matches, mathes_image, mathes_other


def extract_coordinates_and_label(ref_text, image_width, image_height):
    try:
        label_type = ref_text[1]
        # 更安全：用 literal_eval 解析坐标
        cor_list = ast.literal_eval(ref_text[2])
    except Exception as e:
        print(e)
        return None
    return (label_type, cor_list)


def draw_bounding_boxes(image, refs, images_save_dir):
    image_width, image_height = image.size
    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)

    overlay = Image.new('RGBA', img_draw.size, (0, 0, 0, 0))
    draw2 = ImageDraw.Draw(overlay)

    font = ImageFont.load_default()
    img_idx = 0

    for i, ref in enumerate(refs):
        try:
            result = extract_coordinates_and_label(ref, image_width, image_height)
            if result:
                label_type, points_list = result

                color = (np.random.randint(0, 200), np.random.randint(0, 200), np.random.randint(0, 255))
                color_a = color + (20,)
                for points in points_list:
                    x1, y1, x2, y2 = points

                    x1 = int(x1 / 999 * image_width)
                    y1 = int(y1 / 999 * image_height)
                    x2 = int(x2 / 999 * image_width)
                    y2 = int(y2 / 999 * image_height)

                    if label_type == 'image':
                        try:
                            cropped = image.crop((x1, y1, x2, y2))
                            os.makedirs(images_save_dir, exist_ok=True)
                            save_path = os.path.join(images_save_dir, f"{img_idx}.jpg")
                            cropped.save(save_path)
                            print(f"[image] 裁剪图已保存: {save_path}")
                        except Exception as e:
                            print(e)
                            pass
                        img_idx += 1

                    try:
                        if label_type == 'title':
                            draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
                            draw2.rectangle([x1, y1, x2, y2], fill=color_a, outline=(0, 0, 0, 0), width=1)
                        else:
                            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                            draw2.rectangle([x1, y1, x2, y2], fill=color_a, outline=(0, 0, 0, 0), width=1)

                        text_x = x1
                        text_y = max(0, y1 - 15)

                        text_bbox = draw.textbbox((0, 0), label_type, font=font)
                        text_width = text_bbox[2] - text_bbox[0]
                        text_height = text_bbox[3] - text_bbox[1]
                        draw.rectangle([text_x, text_y, text_x + text_width, text_y + text_height],
                                       fill=(255, 255, 255, 30))

                        draw.text((text_x, text_y), label_type, font=font, fill=color)
                    except:
                        pass
        except:
            continue
    img_draw.paste(overlay, (0, 0), overlay)
    return img_draw


def process_image_with_refs(image, ref_texts, images_save_dir):
    result_image = draw_bounding_boxes(image, ref_texts, images_save_dir)
    return result_image


def to_plain_text(model_output: str) -> str:
    """
    将模型输出清理为纯文本：
    - 移除 <|...|> 这类特殊标记
    - 移除 <|ref|>...<|/ref|><|det|>...<|/det|> 整块结构
    - 去掉多余空行与收尾空白
    """
    # 去掉 ref/det 结构块
    text = re.sub(r'<\|ref\|>.*?<\|/ref\|><\|det\|>.*?<\|/det\|>', ' ', model_output, flags=re.DOTALL)
    # 去掉其他 <|...|> 标记
    text = re.sub(r'<\|.*?\|>', ' ', text)
    # 去掉多余空白行
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]
    return "\n".join(lines).strip()


def normalize_human_readable(text: str) -> str:
    """
    将清洗后的纯文本进一步规范为人类可读格式：
    - 去掉 LaTeX 包裹符号 \( \) \[ \] $ $
    - 将常见 LaTeX 符号替换为对应 Unicode
    - 将 \\frac{a}{b} 转为 a/b
    - 去除 Markdown 加粗、分隔线、冗余修饰
    - 规整选项与空白
    """
    t = text

    # 1) 去掉 LaTeX 数学包裹符号与多余反斜杠
    t = re.sub(r'\\\(|\\\)', '', t)  # \( \)
    t = re.sub(r'\\\[|\\\]', '', t)  # \[ \]
    t = re.sub(r'\$(.*?)\$', r'\1', t)  # $...$
    t = t.replace('\\\\', '\\')  # 双反斜杠规整

    # 2) 替换常见 LaTeX 控制序列为 Unicode
    latex_map = {
        r'\leq': '≤', r'\geq': '≥', r'\times': '×', r'\div': '÷',
        r'\pm': '±', r'\cdot': '·', r'\neq': '≠', r'\approx': '≈',
        r'\infty': '∞', r'\to': '→', r'\Leftarrow': '⇐', r'\Rightarrow': '⇒',
        r'\ldots': '…', r'\cdots': '⋯'
    }
    for k, v in latex_map.items():
        t = t.replace(k, v)

    # 3) 处理 \frac{a}{b} => a/b（迭代几次以覆盖简单嵌套）
    def frac_to_plain(m):
        num = m.group(1).strip()
        den = m.group(2).strip()
        return f"{num}/{den}"

    for _ in range(3):
        t = re.sub(r'\\frac\s*{\s*([^{}]+?)\s*}\s*{\s*([^{}]+?)\s*}', frac_to_plain, t)

    # 4) 去掉 Markdown 装饰与分隔
    t = re.sub(r'\*\*(.*?)\*\*', r'\1', t)  # **加粗**
    t = re.sub(r'^-{3,}\s*$', '', t, flags=re.MULTILINE)  # --- 分隔线
    t = re.sub(r'^={3,}\s*$', '', t, flags=re.MULTILINE)  # === 分隔线
    t = re.sub(r'^\s*#+\s*', '', t, flags=re.MULTILINE)  # # 标题前缀

    # 5) 选项规整：确保 A. B. C. D. 前有换行、后有空格
    t = re.sub(r'\s*([A-D])\.\s*', r'\n\1. ', t)

    # 6) 合理合并多余空白
    t = re.sub(r'\n{3,}', '\n\n', t)  # 多空行压成最多 2 行
    t = re.sub(r'[ \t]{2,}', ' ', t)  # 行内多空格压缩

    # 7) 去除首尾空白与统一 Unicode 归一化
    t = unicodedata.normalize('NFKC', t.strip())
    return t


async def stream_generate(image=None, prompt='', skip_special_tokens=False):
    engine_args = AsyncEngineArgs(
        model=MODEL_PATH,
        hf_overrides={"architectures": ["DeepseekOCRForCausalLM"]},
        block_size=256,
        max_model_len=8192,
        enforce_eager=False,
        trust_remote_code=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.75,
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)

    logits_processors = [NoRepeatNGramLogitsProcessor(ngram_size=30, window_size=90,
                                                      whitelist_token_ids={128821, 128822})]  # whitelist: <td>, </td>

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=8192,
        logits_processors=logits_processors,
        skip_special_tokens=skip_special_tokens,
    )

    request_id = f"request-{int(time.time())}"

    printed_length = 0

    if image and '<image>' in prompt:
        request = {
            "prompt": prompt,
            "multi_modal_data": {"image": image}
        }
    elif prompt:
        request = {
            "prompt": prompt
        }
    else:
        assert False, f'prompt is none!!!'
    async for request_output in engine.generate(
            request, sampling_params, request_id
    ):
        if request_output.outputs:
            full_text = request_output.outputs[0].text
            new_text = full_text[printed_length:]
            print(new_text, end='', flush=True)
            printed_length = len(full_text)
            final_output = full_text
    print('\n')

    return final_output


if __name__ == "__main__":
    args = parse_args()

    # 覆盖配置
    input_path = args.input if args.input else INPUT_PATH
    output_base = args.output if args.output else OUTPUT_PATH
    user_prompt = args.prompt  # 先保留用户输入，稍后按模式决定最终 prompt
    crop_mode = args.crop_mode if args.crop_mode is not None else CROP_MODE
    save_results = int(args.save) if args.save is not None else 1
    mode = args.mode  # "text" or "layout"

    # 基于图片名创建独立输出子目录
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    RUN_OUTPUT_DIR = os.path.join(output_base, base_name)
    os.makedirs(RUN_OUTPUT_DIR, exist_ok=True)
    print(f"[dir] 本次输出目录: {RUN_OUTPUT_DIR}")

    # 裁剪小图目录
    images_dir = os.path.join(RUN_OUTPUT_DIR, "images")
    os.makedirs(images_dir, exist_ok=True)

    # 加载图片
    image = load_image(input_path)
    if image is None:
        raise RuntimeError(f"无法加载输入图片: {input_path}")
    image = image.convert('RGB')

    # 按模式决定默认 prompt 与 skip_special_tokens
    if mode == "text":
        # 若用户未自定义 prompt，则使用一个适合 OCR 的默认提示，且必须包含 <image>
        prompt = user_prompt if user_prompt is not None else (
            "<image>\n"
            "你是一名 OCR 系统。请识别图像中的所有可读文本，"
            "按书写/自然阅读顺序输出。仅输出识别到的纯文本，不要添加解释、标签或额外格式。"
        )
        skip_special_tokens = True  # 文本模式下过滤特殊 token 更干净
    else:
        # layout 模式：保持原先配置（需要保留特殊标记以供结构解析）
        prompt = user_prompt if user_prompt is not None else PROMPT
        skip_special_tokens = False

    # 准备图像特征（需要 <image>）
    if '<image>' in prompt:
        image_features = DeepseekOCRProcessor().tokenize_with_images(
            images=[image], bos=True, eos=True, cropping=crop_mode
        )
    else:
        image_features = ''

    # 异步推理
    result_out = asyncio.run(stream_generate(image_features, prompt, skip_special_tokens=skip_special_tokens))

    # 完整打印模型输出
    print("\n" + "=" * 20 + " 模型完整输出 " + "=" * 20)
    print(result_out)
    print("=" * 54 + "\n")

    if mode == "text":
        # 提取纯文本并转为人类可读
        plain_text = to_plain_text(result_out)
        readable_text = normalize_human_readable(plain_text)

        txt_path = os.path.join(RUN_OUTPUT_DIR, "result.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(readable_text)
        print(f"[file] 纯文本 OCR 结果已保存: {txt_path}")

        if '<image>' not in prompt:
            print("[warn] 当前 PROMPT 不包含 <image>，可能未进行图像相关处理。")

    else:
        # layout 模式：沿用原逻辑（mmd、裁剪、绘框、几何等）
        if save_results and '<image>' in prompt:
            print('=' * 15 + ' 保存结果(版面解析) ' + '=' * 15)

            image_draw = image.copy()
            outputs = result_out

            # 保存原始 MMD 到子目录
            result_ori_path = os.path.join(RUN_OUTPUT_DIR, 'result_ori.mmd')
            with open(result_ori_path, 'w', encoding='utf-8') as afile:
                afile.write(outputs)
            print(f"[file] 原始结果已保存: {result_ori_path}")

            # 正则匹配与绘制（裁剪图保存到 images_dir）
            matches_ref, matches_images, mathes_other = re_match(outputs)
            result = process_image_with_refs(image_draw, matches_ref, images_dir)

            # 替换 image/other 标记（保持相对路径 images/idx.jpg）
            for idx, a_match_image in enumerate(tqdm(matches_images, desc="image")):
                outputs = outputs.replace(a_match_image, f'![](images/{idx}.jpg)\n')

            for idx, a_match_other in enumerate(tqdm(mathes_other, desc="other")):
                outputs = outputs.replace(a_match_other, '').replace('\\coloneqq', ':=').replace('\\eqqcolon', '=:')

            # 保存处理后的 MMD 到子目录
            result_mmd_path = os.path.join(RUN_OUTPUT_DIR, 'result.mmd')
            with open(result_mmd_path, 'w', encoding='utf-8') as afile:
                afile.write(outputs)
            print(f"[file] 转换后结果已保存: {result_mmd_path}")

            # 可选几何绘图到子目录（依赖输出格式）
            if 'line_type' in outputs:
                import matplotlib.pyplot as plt
                from matplotlib.patches import Circle

                lines = eval(outputs)['Line']['line']
                line_type = eval(outputs)['Line']['line_type']
                endpoints = eval(outputs)['Line']['line_endpoint']

                fig, ax = plt.subplots(figsize=(3, 3), dpi=200)
                ax.set_xlim(-15, 15)
                ax.set_ylim(-15, 15)

                for idx, line in enumerate(lines):
                    try:
                        p0 = eval(line.split(' -- ')[0])
                        p1 = eval(line.split(' -- ')[-1])

                        if line_type[idx] == '--':
                            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], linewidth=0.8, color='k')
                        else:
                            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], linewidth=0.8, color='k')

                        ax.scatter(p0[0], p0[1], s=5, color='k')
                        ax.scatter(p1[0], p1[1], s=5, color='k')
                    except:
                        pass

                for endpoint in endpoints:
                    label = endpoint.split(': ')[0]
                    (x, y) = eval(endpoint.split(': ')[1])
                    ax.annotate(label, (x, y), xytext=(1, 1), textcoords='offset points',
                                fontsize=5, fontweight='light')

                try:
                    if 'Circle' in eval(outputs).keys():
                        circle_centers = eval(outputs)['Circle']['circle_center']
                        radius = eval(outputs)['Circle']['radius']

                        for center, r in zip(circle_centers, radius):
                            center = eval(center.split(': ')[1])
                            circle = Circle(center, radius=r, fill=False, edgecolor='black', linewidth=0.8)
                            ax.add_patch(circle)
                except:
                    pass

                geo_path = os.path.join(RUN_OUTPUT_DIR, 'geo.jpg')
                plt.savefig(geo_path)
                plt.close()
                print(f"[file] 几何图已保存: {geo_path}")

            # 保存带框的结果图到子目录
            result_img_path = os.path.join(RUN_OUTPUT_DIR, 'result_with_boxes.jpg')
            result.save(result_img_path)
            print(f"[file] 标注图已保存: {result_img_path}")
        else:
            if '<image>' not in prompt:
                print("[warn] 当前 PROMPT 不包含 <image>，未进行图像相关处理与裁剪导出。")
