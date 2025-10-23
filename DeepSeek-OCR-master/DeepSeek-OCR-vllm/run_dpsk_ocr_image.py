python
import asyncio
import re
import os
import time
import argparse

import torch

# 针对 CUDA 11.8 的 Linux 设定（在 Windows 上通常不生效，可忽略）
if torch.version.cuda == '11.8':
    os.environ["TRITON_PTXAS_PATH"] = "/usr/local/cuda-11.8/bin/ptxas"

# vLLM 相关环境变量
os.environ['VLLM_USE_V1'] = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

from vllm import AsyncLLMEngine, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.model_executor.models.registry import ModelRegistry

from PIL import Image, ImageDraw, ImageFont, ImageOps
import numpy as np
from tqdm import tqdm

from deepseek_ocr import DeepseekOCRForCausalLM
from process.ngram_norepeat import NoRepeatNGramLogitsProcessor
from process.image_process import DeepseekOCRProcessor
from config import MODEL_PATH, INPUT_PATH, OUTPUT_PATH, PROMPT, CROP_MODE

# 注册自定义模型
ModelRegistry.register_model("DeepseekOCRForCausalLM", DeepseekOCRForCausalLM)


def parse_args():
    parser = argparse.ArgumentParser(description="Run DeepSeek OCR inference (vLLM).")
    parser.add_argument("-i", "--input", type=str, required=False,
                        help="输入图片文件路径（覆盖 config.INPUT_PATH）")
    parser.add_argument("-o", "--output-dir", type=str, required=False,
                        help="导出目录（覆盖 config.OUTPUT_PATH）")
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
    # 匹配 <|ref|>xxx<|/ref|><|det|>[...]<|/det|> 结构
    pattern = r'(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)'
    matches = re.findall(pattern, text, re.DOTALL)

    mathes_image = []
    mathes_other = []
    for a_match in matches:
        if '<|ref|>image<|/ref|>' in a_match[0].lower():
            mathes_image.append(a_match[0])
        else:
            mathes_other.append(a_match[0])
    return matches, mathes_image, mathes_other


def extract_coordinates_and_label(ref_text, image_width, image_height):
    try:
        label_type = ref_text[1]
        cor_list = eval(ref_text[2])  # 注意：模型输出应尽量改为 JSON，使用 json.loads 更安全
    except Exception as e:
        print(e)
        return None
    return (label_type, cor_list)


def _normalize_and_clip_box(x1, y1, x2, y2, image_width, image_height):
    # 0..999 归一化坐标 -> 像素坐标
    x1 = int(x1 / 999.0 * image_width)
    y1 = int(y1 / 999.0 * image_height)
    x2 = int(x2 / 999.0 * image_width)
    y2 = int(y2 / 999.0 * image_height)

    # 顺序纠正，确保左上到右下
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1

    # 边界裁剪
    x1 = max(0, min(x1, image_width - 1))
    x2 = max(0, min(x2, image_width))
    y1 = max(0, min(y1, image_height - 1))
    y2 = max(0, min(y2, image_height))

    # 避免 0 尺寸
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def draw_bounding_boxes(image, refs, output_dir):
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
            if not result:
                continue
            label_type, points_list = result

            label_type_lower = (label_type or "").strip().lower()

            color = (np.random.randint(0, 200), np.random.randint(0, 200), np.random.randint(0, 255))
            color_a = color + (20,)

            for points in points_list:
                try:
                    x1, y1, x2, y2 = points
                except Exception:
                    continue

                box = _normalize_and_clip_box(x1, y1, x2, y2, image_width, image_height)
                if box is None:
                    # 无效框，跳过
                    continue

                bx1, by1, bx2, by2 = box

                # 保存裁剪子图（仅当标签为 image）
                if label_type_lower == 'image':
                    try:
                        cropped = image.crop((bx1, by1, bx2, by2))
                        save_path = os.path.join(output_dir, "images", f"{img_idx}.jpg")
                        cropped.save(save_path)
                        img_idx += 1  # 仅在保存成功后递增
                    except Exception as e:
                        print(f"Crop/Save error: {e}")

                # 绘制框与标签
                try:
                    if label_type_lower == 'title':
                        draw.rectangle([bx1, by1, bx2, by2], outline=color, width=4)
                        draw2.rectangle([bx1, by1, bx2, by2], fill=color_a, outline=(0, 0, 0, 0), width=1)
                    else:
                        draw.rectangle([bx1, by1, bx2, by2], outline=color, width=2)
                        draw2.rectangle([bx1, by1, bx2, by2], fill=color_a, outline=(0, 0, 0, 0), width=1)

                    text_x = bx1
                    text_y = max(0, by1 - 15)
                    text_bbox = draw.textbbox((0, 0), label_type, font=font)
                    text_width = text_bbox[2] - text_bbox[0]
                    text_height = text_bbox[3] - text_bbox[1]
                    draw.rectangle([text_x, text_y, text_x + text_width, text_y + text_height],
                                   fill=(255, 255, 255, 30))
                    draw.text((text_x, text_y), label_type, font=font, fill=color)
                except Exception:
                    pass

        except Exception:
            continue

    img_draw.paste(overlay, (0, 0), overlay)
    return img_draw


def process_image_with_refs(image, ref_texts, output_dir):
    result_image = draw_bounding_boxes(image, ref_texts, output_dir)
    return result_image


async def stream_generate(image=None, prompt=''):
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
        skip_special_tokens=False,
        # ignore_eos=False,
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

    final_output = ""
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

    # 最终输入/输出路径（命令行优先）
    final_input_path = args.input if args.input else INPUT_PATH
    final_output_dir = args.output_dir if args.output_dir else OUTPUT_PATH

    os.makedirs(final_output_dir, exist_ok=True)
    os.makedirs(os.path.join(final_output_dir, 'images'), exist_ok=True)

    # 读取图像
    image = load_image(final_input_path)
    if image is None:
        raise FileNotFoundError(f"无法打开输入图片: {final_input_path}")
    image = image.convert('RGB')

    # 基于输入图片名构造 Markdown 文件名
    base_name = os.path.splitext(os.path.basename(final_input_path))[0]
    md_path_ori = os.path.join(final_output_dir, f"{base_name}_ori.md")
    md_path_clean = os.path.join(final_output_dir, f"{base_name}.md")

    # 处理多模态输入
    if '<image>' in PROMPT:
        image_features = DeepseekOCRProcessor().tokenize_with_images(
            images=[image], bos=True, eos=True, cropping=CROP_MODE
        )
    else:
        image_features = ''

    prompt = PROMPT

    # 推理
    result_out = asyncio.run(stream_generate(image_features, prompt))

    save_results = 1

    if save_results and '<image>' in prompt:
        print('=' * 15 + 'save results:' + '=' * 15)

        image_draw = image.copy()
        outputs = result_out

        # 保存原始输出（.md）
        with open(md_path_ori, 'w', encoding='utf-8') as afile:
            afile.write(outputs)

        # 解析标签并绘制/裁剪
        matches_ref, matches_images, mathes_other = re_match(outputs)
        result = process_image_with_refs(image_draw, matches_ref, final_output_dir)

        # 将 image 类型标签替换为 Markdown 图片引用
        for idx, a_match_image in enumerate(tqdm(matches_images, desc="image")):
            outputs = outputs.replace(a_match_image, f'![](images/{idx}.jpg)\n')

        # 清理其他标签与 LaTeX 特殊符号
        for idx, a_match_other in enumerate(tqdm(mathes_other, desc="other")):
            outputs = outputs.replace(a_match_other, '').replace('\\\\coloneqq', ':=').replace('\\\\eqqcolon', '=:')
        # 保存清洗后的 Markdown
        with open(md_path_clean, 'w', encoding='utf-8') as afile:
            afile.write(outputs)

        # 几何绘制（可选）
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

            plt.savefig(os.path.join(final_output_dir, 'geo.jpg'))
            plt.close()

        # 保存带框图
        result.save(os.path.join(final_output_dir, 'result_with_boxes.jpg'))
