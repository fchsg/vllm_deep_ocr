import os
import re
import ast
import time
import argparse
import unicodedata
from typing import List, Tuple

import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps
import numpy as np
from tqdm import tqdm

from vllm import LLM, SamplingParams

# 你本地的配置（保留原有 config 引入，也可以通过命令行覆盖）
from config import MODEL_PATH, OUTPUT_PATH as CFG_OUTPUT_PATH, PROMPT as CFG_PROMPT

# 可选：根据 CUDA 版本做 triton 配置（与原脚本一致）
if hasattr(torch, "version") and getattr(torch.version, "cuda", None) == "11.8":
    os.environ["TRITON_PTXAS_PATH"] = "/usr/local/cuda-11.8/bin/ptxas"

os.environ['VLLM_USE_V1'] = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")


# -----------------------------
# 工具函数
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="DeepSeek-OCR vLLM 批量图像文本与区域提取")
    parser.add_argument("-i", "--input", required=True, type=str, help="输入文件或目录路径（图片或包含图片的目录）")
    parser.add_argument("-o", "--output", type=str, default=CFG_OUTPUT_PATH, help="输出根目录，默认读取 config.OUTPUT_PATH")
    parser.add_argument("-m", "--model", type=str, default=MODEL_PATH, help="模型路径或名称，默认读取 config.MODEL_PATH")
    parser.add_argument("-p", "--prompt", type=str, default=None, help="自定义提示词。不传则使用默认：'<image>\\nFree OCR.'")
    parser.add_argument("--temperature", type=float, default=0.0, help="采样温度，默认 0")
    parser.add_argument("--max-tokens", type=int, default=8192, help="最大生成 token 数，默认 8192")
    parser.add_argument("--skip-special-tokens", action="store_true",
                        help="跳过特殊 Token，清爽文本（注意：可能影响版面结构解析）")
    return parser.parse_args()


def find_images(input_path: str) -> List[str]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    paths = []
    if os.path.isfile(input_path):
        if os.path.splitext(input_path.lower())[1] in exts:
            paths.append(os.path.abspath(input_path))
    elif os.path.isdir(input_path):
        for name in os.listdir(input_path):
            fp = os.path.join(input_path, name)
            if os.path.isfile(fp) and os.path.splitext(name.lower())[1] in exts:
                paths.append(os.path.abspath(fp))
    else:
        raise FileNotFoundError(f"输入路径不存在: {input_path}")
    if not paths:
        raise RuntimeError(f"在 {input_path} 未找到图片")
    return sorted(paths)


def load_image(image_path: str) -> Image.Image:
    try:
        image = Image.open(image_path)
        corrected_image = ImageOps.exif_transpose(image)
        return corrected_image.convert("RGB")
    except Exception as e:
        print(f"[warn] 加载图片失败({image_path}): {e}")
        img = Image.open(image_path).convert("RGB")
        return img


def re_match(text: str):
    # 匹配 (<|ref|>...</|ref|><|det|>...</|/det|>)
    pattern = r'(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)'
    matches = re.findall(pattern, text, re.DOTALL)
    matches_image = []
    matches_other = []
    for a_match in matches:
        if '<|ref|>image<|/ref|>' in a_match[0]:
            matches_image.append(a_match[0])
        else:
            matches_other.append(a_match[0])
    return matches, matches_image, matches_other


def extract_coordinates_and_label(ref_text: Tuple[str, str, str], image_width: int, image_height: int):
    try:
        label_type = ref_text[1]
        # 安全解析坐标列表
        cor_list = ast.literal_eval(ref_text[2])
    except Exception as e:
        print(f"[warn] 解析坐标失败: {e}")
        return None
    return (label_type, cor_list)


def draw_bounding_boxes_and_crop(image: Image.Image, refs, images_save_dir: str) -> Image.Image:
    image_width, image_height = image.size
    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)
    overlay = Image.new('RGBA', img_draw.size, (0, 0, 0, 0))
    draw2 = ImageDraw.Draw(overlay)
    font = ImageFont.load_default()
    img_idx = 0

    os.makedirs(images_save_dir, exist_ok=True)

    for i, ref in enumerate(refs):
        try:
            result = extract_coordinates_and_label(ref, image_width, image_height)
            if not result:
                continue
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
                        save_path = os.path.join(images_save_dir, f"{img_idx}.jpg")
                        cropped.save(save_path)
                        print(f"[image] 裁剪图已保存: {save_path}")
                    except Exception as e:
                        print(f"[warn] 保存裁剪失败: {e}")
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
                except Exception:
                    pass
        except Exception:
            continue
    img_draw.paste(overlay, (0, 0), overlay)
    return img_draw


def to_plain_text(model_output: str) -> str:
    r"""
    清理模型输出为纯文本：
    - 移除 <|...|> 特殊标记 与 ref/det 结构
    - 去掉多余空白行
    """
    text = re.sub(r'<\|ref\|>.*?<\|/ref\|><\|det\|>.*?<\|/det\|>', ' ', model_output, flags=re.DOTALL)
    text = re.sub(r'<\|.*?\|>', ' ', text)
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]
    return "\n".join(lines).strip()


def normalize_human_readable(text: str) -> str:
    r"""
    进一步规范为人类可读格式：
    - 去掉 LaTeX 包裹符号 \( \) \[ \] $ $
    - 替换常用 LaTeX 控制序列为 Unicode
    - \frac{a}{b} => a/b
    - 去掉 Markdown 粗体/分隔线，规整选项
    """
    t = text
    t = re.sub(r'\\\(|\\\)', '', t)      # \( \)
    t = re.sub(r'\\\[|\\\]', '', t)      # \[ \]
    t = re.sub(r'\$(.*?)\$', r'\1', t)   # $...$
    t = t.replace('\\\\', '\\')

    latex_map = {
        r'\leq': '≤', r'\geq': '≥', r'\times': '×', r'\div': '÷',
        r'\pm': '±', r'\cdot': '·', r'\neq': '≠', r'\approx': '≈',
        r'\infty': '∞', r'\to': '→', r'\Leftarrow': '⇐', r'\Rightarrow': '⇒',
        r'\ldots': '…', r'\cdots': '⋯'
    }
    for k, v in latex_map.items():
        t = t.replace(k, v)

    def frac_to_plain(m):
        num = m.group(1).strip()
        den = m.group(2).strip()
        return f"{num}/{den}"
    for _ in range(3):
        t = re.sub(r'\\frac\s*{\s*([^{}]+?)\s*}\s*{\s*([^{}]+?)\s*}', frac_to_plain, t)

    t = re.sub(r'\*\*(.*?)\*\*', r'\1', t)
    t = re.sub(r'^-{3,}\s*$', '', t, flags=re.MULTILINE)
    t = re.sub(r'^={3,}\s*$', '', t, flags=re.MULTILINE)
    t = re.sub(r'^\s*#+\s*', '', t, flags=re.MULTILINE)

    t = re.sub(r'\s*([A-D])\.\s*', r'\n\1. ', t)

    t = re.sub(r'\n{3,}', '\n\n', t)
    t = re.sub(r'[ \t]{2,}', ' ', t)

    t = unicodedata.normalize('NFKC', t.strip())
    return t


def build_llm(model_path: str) -> LLM:
    # 关键修改：不再传 logits_processors，避免对内部模块路径的依赖
    llm = LLM(
        model=model_path,
        trust_remote_code=True,  # deepseek 模型通常需要
        # enable_prefix_caching=False,
        # mm_processor_cache_gb=0
    )
    return llm


def build_sampling_params(temperature: float, max_tokens: int, skip_special_tokens: bool) -> SamplingParams:
    sp = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        skip_special_tokens=skip_special_tokens,
        extra_args=dict(
            ngram_size=30,
            window_size=90,
            whitelist_token_ids={128821, 128822},  # whitelist: <td>, </td>
        ),
    )
    return sp


def main():
    args = parse_args()

    # 收集图片
    image_paths = find_images(args.input)

    # 输出根目录
    output_root = os.path.abspath(args.output)
    os.makedirs(output_root, exist_ok=True)

    # Prompt
    prompt = args.prompt if args.prompt is not None else "<image>\nFree OCR."

    # 构建 LLM 与采样参数
    llm = build_llm(args.model)
    sampling_param = build_sampling_params(args.temperature, args.max_tokens, args.skip_special_tokens)

    # 构造批量输入
    model_input = []
    pil_images = []
    for img_path in image_paths:
        img = load_image(img_path)
        pil_images.append(img)
        model_input.append({
            "prompt": prompt,
            "multi_modal_data": {"image": img}
        })

    # 推理
    print(f"[info] 开始生成，批次数: {len(model_input)}")
    outputs = llm.generate(model_input, sampling_param)

    # 逐图片保存结果
    for idx, output in enumerate(outputs):
        full_text = output.outputs[0].text if output.outputs and output.outputs[0].text is not None else ""
        img_path = image_paths[idx]
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        run_dir = os.path.join(output_root, base_name)
        images_dir = os.path.join(run_dir, "images")
        os.makedirs(run_dir, exist_ok=True)
        os.makedirs(images_dir, exist_ok=True)

        print("\n" + "=" * 20 + f" {base_name} - 模型完整输出 " + "=" * 20)
        print(full_text)
        print("=" * 54 + "\n")

        # 保存原始输出
        raw_path = os.path.join(run_dir, "result_raw.txt")
        with open(raw_path, "w", encoding="utf-8") as f:
            f.write(full_text)
        print(f"[file] 原始输出已保存: {raw_path}")

        # 人类可读文本
        plain_text = to_plain_text(full_text)
        readable_text = normalize_human_readable(plain_text)
        txt_path = os.path.join(run_dir, "result.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(readable_text)
        print(f"[file] 纯文本 OCR 结果已保存: {txt_path}")

        # 解析结构并裁剪图片、绘制标注、生成 mmd
        img = pil_images[idx]
        matches_ref, matches_images, matches_other = re_match(full_text)
        if matches_ref:
            result_with_boxes = draw_bounding_boxes_and_crop(img, matches_ref, images_dir)
            result_img_path = os.path.join(run_dir, 'result_with_boxes.jpg')
            result_with_boxes.save(result_img_path)
            print(f"[file] 标注图已保存: {result_img_path}")

            # 生成 mmd：替换 image 区块、移除其他区块
            mmd_out = full_text
            for k, a_match_image in enumerate(tqdm(matches_images, desc=f"{base_name} images")):
                mmd_out = mmd_out.replace(a_match_image, f'![](images/{k}.jpg)\n')
            for a_match_other in tqdm(matches_other, desc=f"{base_name} other"):
                mmd_out = mmd_out.replace(a_match_other, '').replace('\\coloneqq', ':=').replace('\\eqqcolon', '=:')
            mmd_path = os.path.join(run_dir, 'result.mmd')
            with open(mmd_path, 'w', encoding='utf-8') as f:
                f.write(mmd_out)
            print(f"[file] MMD 输出已保存: {mmd_path}")
        else:
            print(f"[warn] 未检测到可解析的 <|ref|>/<|det|> 结构，跳过裁剪与标注。")

    print("\n[done] 全部处理完成。")


if __name__ == "__main__":
    main()
