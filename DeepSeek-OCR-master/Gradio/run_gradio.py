#!/usr/bin/env python3
# run_gradio.py
# 依赖: pip install gradio transformers torch pillow pylatexenc

import gradio as gr
from transformers import AutoModel, AutoTokenizer
import torch
import os
from PIL import Image
import tempfile
import shutil
import io
import base64
import re

# pylatexenc 用于 LaTeX -> 可读文本（若未安装回退）
try:
    from pylatexenc.latex2text import LatexNodes2Text
except Exception:
    LatexNodes2Text = None
    print("警告: pylatexenc 未安装，LaTeX 转文本将回退到原文。可执行: pip install pylatexenc")

# 全局模型变量（延迟加载）
model = None
tokenizer = None


def load_model():
    """延迟加载 DeepSeek-OCR 模型与 tokenizer。若无 GPU 回退 CPU。"""
    global model, tokenizer
    if model is None:
        print("Loading DeepSeek-OCR model...")
        model_name = 'deepseek-ai/DeepSeek-OCR'
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModel.from_pretrained(
            model_name,
            _attn_implementation='flash_attention_2',
            trust_remote_code=True,
            use_safetensors=True
        )
        # 尝试 GPU + bfloat16，失败则回退到 CPU + float32
        try:
            model = model.eval().cuda().to(torch.bfloat16)
        except Exception:
            model = model.eval().to(torch.float32)
        print("Model loaded successfully!")
    return model, tokenizer


# -------------------------
# 图像与 base64 编码辅助
# -------------------------
def pil_image_to_base64_datauri(img: Image.Image, max_width=600, quality=85, fmt="JPEG"):
    """
    把 PIL.Image 转为缩放+压缩后的 base64 data URI（JPEG 默认）。
    用于在 Markdown 中嵌入（控制大小）。
    """
    if img is None:
        return None
    try:
        w, h = img.size
    except Exception:
        return None

    if max_width and w > max_width:
        new_w = max_width
        new_h = int(h * (new_w / w))
        img = img.resize((new_w, new_h), Image.LANCZOS)

    img_format = fmt.upper()
    buf = io.BytesIO()
    save_kwargs = {}
    if img_format == "JPEG":
        # 处理透明通道
        if img.mode in ("RGBA", "LA"):
            background = Image.new("RGB", img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[3])
            img_to_save = background
        else:
            img_to_save = img.convert("RGB")
        save_kwargs["quality"] = quality
    else:
        img_to_save = img

    img_to_save.save(buf, format=img_format, **save_kwargs)
    b = buf.getvalue()
    encoded = base64.b64encode(b).decode("ascii")
    mime = "image/jpeg" if img_format == "JPEG" else f"image/{img_format.lower()}"
    return f"data:{mime};base64,{encoded}"


def pil_image_to_base64_datauri_raw(img: Image.Image, fmt="PNG"):
    """
    把 PIL.Image 以原始像素（或指定 fmt）编码为 base64 data URI（不缩放，用于 preserve_size）。
    默认 PNG 无损。
    """
    if img is None:
        return None
    try:
        buf = io.BytesIO()
        img_to_save = img
        if fmt.upper() == "JPEG":
            if img.mode in ("RGBA", "LA"):
                background = Image.new("RGB", img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[3])
                img_to_save = background
            else:
                img_to_save = img.convert("RGB")
            img_to_save.save(buf, format="JPEG", quality=95)
        else:
            img_to_save.save(buf, format=fmt.upper())
        b = buf.getvalue()
        encoded = base64.b64encode(b).decode("ascii")
        mime = "image/png" if fmt.upper() == "PNG" else "image/jpeg"
        return f"data:{mime};base64,{encoded}"
    except Exception:
        return None


def images_to_base64_list(images, max_width=600, quality=85, fmt="JPEG"):
    """把 PIL.Image 列表转为 base64 列表（缩放压缩）。"""
    uris = []
    if not images:
        return uris
    for img in images:
        try:
            if isinstance(img, str) and os.path.exists(img):
                pil_img = Image.open(img).convert("RGB")
            else:
                pil_img = img
            uri = pil_image_to_base64_datauri(pil_img, max_width=max_width, quality=quality, fmt=fmt)
            if uri:
                uris.append(uri)
        except Exception:
            continue
    return uris


# -------------------------
# 收集 temp_dir（模型 output_path）中的图片（碎图）
# -------------------------
def collect_patch_images(output_dir, max_patches=24):
    """
    在 output_dir 中查找可能的碎图（patches/crops）。
    返回 PIL.Image 列表（convert("RGB")），按发现顺序，最多 max_patches。
    """
    if not output_dir or not os.path.exists(output_dir):
        return []

    exts = ('.png', '.jpg', '.jpeg', '.webp')
    keywords = ['patch', 'patches', 'crop', 'crops', 'fragment', 'frag', 'cropbox', 'box', 'crop_img', 'patch_img', 'vis']
    found_paths = []

    # 首先优先找带关键词的文件或目录
    for root, dirs, files in os.walk(output_dir):
        dir_name = os.path.basename(root).lower()
        dir_priority = any(k in dir_name for k in keywords)
        for f in files:
            if f.lower().endswith(exts):
                full = os.path.join(root, f)
                fname = f.lower()
                if dir_priority or any(k in fname for k in keywords):
                    found_paths.append(full)

    # 回退到任意图片
    if not found_paths:
        for root, dirs, files in os.walk(output_dir):
            for f in files:
                if f.lower().endswith(exts):
                    found_paths.append(os.path.join(root, f))

    unique_paths = list(dict.fromkeys(found_paths))
    images = []
    for p in unique_paths:
        try:
            img = Image.open(p).convert("RGB")
            images.append(img)
        except Exception:
            continue
        if len(images) >= max_patches:
            break
    return images


# -------------------------
# LaTeX -> 可读文本
# -------------------------
def latex_to_readable_text(latex_str: str) -> str:
    if not latex_str or not latex_str.strip():
        return latex_str
    if LatexNodes2Text is None:
        return latex_str
    try:
        return LatexNodes2Text().latex_to_text(latex_str)
    except Exception:
        return latex_str


# -------------------------
# 运行模型并收集碎图（仅返回 OCR 文本与 patches）
# -------------------------
def process_image_collect_patches(image, prompt_type, custom_prompt, model_size):
    """
    执行 model.infer 并从 output_path 临时目录收集碎图（patches）。
    返回 (readable_text, patches_list)
    """
    try:
        model, tokenizer = load_model()
        temp_dir = tempfile.mkdtemp()

        # 保存上传图像
        temp_image_path = os.path.join(temp_dir, "input_image.jpg")
        if isinstance(image, str):
            shutil.copy(image, temp_image_path)
        else:
            image.save(temp_image_path)

        # 构造 prompt
        if prompt_type == "Free OCR":
            prompt = "<image>\nFree OCR. "
        elif prompt_type == "Markdown Conversion":
            prompt = "<image>\n<|grounding|>Convert the document to markdown. "
        elif prompt_type == "Custom":
            prompt = f"<image>\n{custom_prompt}"
        else:
            prompt = "<image>\nFree OCR. "

        size_configs = {
            "Tiny": {"base_size": 512, "image_size": 512, "crop_mode": False},
            "Small": {"base_size": 640, "image_size": 640, "crop_mode": False},
            "Base": {"base_size": 1024, "image_size": 1024, "crop_mode": False},
            "Large": {"base_size": 1280, "image_size": 1280, "crop_mode": False},
            "Gundam (Recommended)": {"base_size": 1024, "image_size": 640, "crop_mode": True}
        }
        config = size_configs.get(model_size, size_configs["Gundam (Recommended)"])

        # 捕获 stdout 执行模型推理（model.infer 可能把结果写入 temp_dir）
        import sys
        from io import StringIO
        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()
        try:
            result = model.infer(
                tokenizer,
                prompt=prompt,
                image_file=temp_image_path,
                output_path=temp_dir,
                base_size=config["base_size"],
                image_size=config["image_size"],
                crop_mode=config["crop_mode"],
                save_results=True,
                test_compress=False
            )
        finally:
            sys.stdout = old_stdout

        captured_text = captured_output.getvalue()

        # 优先读取 temp_dir 下的 txt 文件作为 OCR 文本
        ocr_text = ""
        for filename in os.listdir(temp_dir):
            if filename.endswith('.txt'):
                try:
                    with open(os.path.join(temp_dir, filename), 'r', encoding='utf-8') as f:
                        ocr_text += f.read() + "\n"
                except Exception:
                    pass

        # 若没有 txt，则解析 captured_text
        if not ocr_text.strip() and captured_text.strip():
            lines = captured_text.splitlines()
            clean_lines = []
            for line in lines:
                if '<|ref|>' in line or '<|det|>' in line or '<|/ref|>' in line or '<|/det|>' in line:
                    m = re.search(r'<\|/ref\|>(.*?)<\|det\|>', line)
                    if m:
                        clean_lines.append(m.group(1).strip())
                elif line.startswith('=====') or 'BASE:' in line or 'PATCHES:' in line or line.startswith('image:') or line.startswith('other:'):
                    continue
                elif line.strip():
                    clean_lines.append(line.strip())
            ocr_text = "\n".join(clean_lines)

        if not ocr_text.strip() and isinstance(result, str):
            ocr_text = result

        readable = latex_to_readable_text(ocr_text) if ocr_text else ""

        # 收集 patch images（模型在 temp_dir 写出的碎图）
        patches = collect_patch_images(temp_dir, max_patches=32)

        # 清理临时目录（patches 已加载到内存）
        try:
            shutil.rmtree(temp_dir)
        except Exception:
            pass

        return readable if readable.strip() else "No text detected in image.", patches

    except Exception as e:
        import traceback
        msg = f"Error: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        return msg, []


# -------------------------
# 构建 Markdown：只包含碎图（支持 preserve_size）
# -------------------------
def build_markdown_from_patches(readable_text: str, patches_images=None, patches_base64=None,
                                embed_base64=True, preserve_size=False, patch_max_width=600, quality=85):
    """
    仅把 patches 嵌入到 Markdown 中；不会包含过程图或原图。
    preserve_size=True 时嵌入原始像素（PNG），否则缩放后嵌入（JPEG）。
    """
    # 准备 base64 列表
    if patches_base64 is None and patches_images:
        patches_base64 = []
        if embed_base64:
            if preserve_size:
                for img in patches_images:
                    uri = pil_image_to_base64_datauri_raw(img, fmt="PNG")
                    if uri:
                        patches_base64.append(uri)
            else:
                patches_base64 = images_to_base64_list(patches_images, max_width=patch_max_width, quality=quality)
        else:
            patches_base64 = None

    md_lines = []
    md_lines.append("# OCR 结果\n")
    md_lines.append("## 文本\n")
    md_lines.append(readable_text or "*未识别到文本*")
    md_lines.append("\n---\n")
    md_lines.append("## 识别为图片的碎图（仅碎图）\n")

    count = len(patches_base64) if patches_base64 is not None else (len(patches_images) if patches_images else 0)
    md_lines.append(f"_共识别到 {count} 张碎图_\n\n")

    if count == 0:
        md_lines.append("_无碎图_\n")
    else:
        for idx in range(1, count + 1):
            md_lines.append(f"### 碎图 {idx}\n")
            if patches_base64 is not None:
                uri = patches_base64[idx - 1]
                md_lines.append(f"![patch_{idx}]({uri})\n")
            else:
                # 不嵌入 -> 相对文件名（导出时会保存）
                md_lines.append(f"![patch_{idx}](patch_{idx}.png)\n")
            md_lines.append("\n")

    return "\n".join(md_lines)


# -------------------------
# 导出 Markdown：当 embed=False 时保存原始尺寸图片文件；当 embed=True 且 preserve_size=True 则 MD 已包含原始
# -------------------------
def export_markdown_with_patches(markdown_text: str, patches_images, embed_base64=True, preserve_size=False, quality=95):
    try:
        temp_dir = tempfile.mkdtemp()
        md_path = os.path.join(temp_dir, "result.md")

        if not embed_base64:
            # 保存 patches 为 PNG（原始像素以尽量保持一致）
            for idx, p in enumerate(patches_images or [], start=1):
                try:
                    out_path = os.path.join(temp_dir, f"patch_{idx}.png")
                    p.save(out_path, format="PNG")
                except Exception:
                    try:
                        p.save(os.path.join(temp_dir, f"patch_{idx}.jpg"), quality=quality)
                    except Exception:
                        pass

        with open(md_path, "w", encoding="utf-8") as f:
            f.write(markdown_text or "")

        return md_path
    except Exception as e:
        print(f"导出失败: {e}")
        return None


# -------------------------
# Gradio 界面（简洁、左输入右结果）
# -------------------------
def create_demo():
    with gr.Blocks(title="DeepSeek-OCR (patches only)", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🔍 DeepSeek-OCR - 导出碎图为 Markdown\n上传图片 -> OCR -> 返回可读文本与识别为图片的碎图（patches）。生成 Markdown 时只包含碎图，支持嵌入 base64 与保持原始尺寸选项。")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📤 Input & Settings")
                image_input = gr.Image(label="Upload Image", type="pil", sources=["upload", "clipboard"])
                prompt_type = gr.Radio(choices=["Free OCR", "Markdown Conversion", "Custom"], value="Markdown Conversion", label="Prompt Type")
                custom_prompt = gr.Textbox(label="Custom Prompt (if selected)", placeholder="Enter custom prompt...", lines=2, visible=False)
                model_size = gr.Radio(choices=["Tiny", "Small", "Base", "Large", "Gundam (Recommended)"], value="Gundam (Recommended)", label="Model Size")
                process_btn = gr.Button("🚀 Process Image", variant="primary")

                def update_prompt_visibility(choice):
                    return gr.update(visible=(choice == "Custom"))
                prompt_type.change(fn=update_prompt_visibility, inputs=[prompt_type], outputs=[custom_prompt])

            with gr.Column(scale=1):
                gr.Markdown("### 📄 Results")
                output_text = gr.Textbox(label="Extracted Text (readable)", lines=18, max_lines=200, show_copy_button=True)
                patches_gallery = gr.Gallery(label="识别为图片的碎图 (patches)", columns=6, type="pil")

                gr.Markdown("### 📝 Markdown & Export")
                readable_toggle = gr.Checkbox(label="将 LaTeX 转换为可读文本（pylatexenc）", value=True)
                embed_toggle = gr.Checkbox(label="在 Markdown 中嵌入图片（Base64）", value=True)
                preserve_size_checkbox = gr.Checkbox(label="保持碎图原始尺寸（导出/嵌入时不缩放）", value=False)
                generate_md_btn = gr.Button("📝 生成 Markdown")
                md_preview = gr.Markdown(label="Markdown 预览", value="")
                export_md_btn = gr.Button("💾 导出 Markdown (.md)")
                md_file = gr.File(label="下载生成的 Markdown 文件", interactive=False)

        # 处理 -> 返回可读文本 + patches 列表（用于 Gallery）
        process_btn.click(
            fn=process_image_collect_patches,
            inputs=[image_input, prompt_type, custom_prompt, model_size],
            outputs=[output_text, patches_gallery]
        )

        # 生成 Markdown：只包含碎图（根据 embed / preserve_size）
        def on_generate_md(text_result, patches_list, use_readable, embed_base64, preserve_size):
            readable = text_result or ""
            patches_images = patches_list or []
            md = build_markdown_from_patches(readable, patches_images=patches_images, patches_base64=None,
                                            embed_base64=embed_base64, preserve_size=preserve_size,
                                            patch_max_width=600, quality=85)
            return md

        generate_md_btn.click(
            fn=on_generate_md,
            inputs=[output_text, patches_gallery, readable_toggle, embed_toggle, preserve_size_checkbox],
            outputs=[md_preview]
        )

        # 导出 Markdown：写入 temp dir 并返回 md 路径供下载
        def on_export_md(md_str, patches_list, embed_base64, preserve_size):
            patches_images = patches_list or []
            md_path = export_markdown_with_patches(md_str, patches_images, embed_base64=embed_base64, preserve_size=preserve_size)
            return md_path

        export_md_btn.click(
            fn=on_export_md,
            inputs=[md_preview, patches_gallery, embed_toggle, preserve_size_checkbox],
            outputs=[md_file]
        )

    return demo


if __name__ == "__main__":
    demo = create_demo()
    demo.launch(server_name="0.0.0.0", server_port=2714, share=False)
