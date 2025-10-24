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
import mimetypes

# pylatexenc 用于 LaTeX -> 可读文本（若未安装会回退）
try:
    from pylatexenc.latex2text import LatexNodes2Text
except Exception:
    LatexNodes2Text = None
    print("警告: pylatexenc 未安装，LaTeX 转文本将回退到原文。可执行: pip install pylatexenc")

# 全局模型与 tokenizer（延迟加载）
model = None
tokenizer = None


def load_model():
    """延迟加载模型，优先使用 GPU:bfloat16，否则回退 CPU:float32"""
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
        # 尝试使用 GPU + bfloat16
        try:
            model = model.eval().cuda().to(torch.bfloat16)
        except Exception:
            model = model.eval().to(torch.float32)
        print("Model loaded successfully!")
    return model, tokenizer


# -------------------------
# 辅助：读取文件为 base64 data URI，尽量保留原始格式
# -------------------------
def file_to_data_uri(file_path):
    """
    从磁盘文件读取字节并生成 data URI，保持原始文件的 MIME 类型（根据扩展名推断）。
    返回 data URI 字符串或 None。
    """
    try:
        with open(file_path, "rb") as f:
            b = f.read()
        mime, _ = mimetypes.guess_type(file_path)
        if not mime:
            # fallback to jpeg
            mime = "image/jpeg"
        encoded = base64.b64encode(b).decode("ascii")
        return f"data:{mime};base64,{encoded}"
    except Exception:
        return None


def pil_to_data_uri_preserve_format(pil_img: Image.Image, original_format=None, preserve_size=False, max_width=None, quality=85):
    """
    将 PIL.Image 编码为 data URI。
    - original_format: 如果提供（'PNG','JPEG'等），尽量使用该格式编码；否则默认 PNG（无损）。
    - preserve_size: 如果 True，不进行缩放；否则可按 max_width 缩放
    - max_width: 控制缩放宽度（当 preserve_size=False 且 max_width 不为 None 时生效）
    """
    if pil_img is None:
        return None
    img = pil_img
    try:
        w, h = img.size
    except Exception:
        return None

    if not preserve_size and max_width and w > max_width:
        new_w = max_width
        new_h = int(h * (new_w / w))
        img = img.resize((new_w, new_h), Image.LANCZOS)

    fmt = (original_format or "PNG").upper()
    buf = io.BytesIO()
    try:
        if fmt in ("JPG", "JPEG"):
            # JPEG 保存需要 RGB
            if img.mode in ("RGBA", "LA"):
                background = Image.new("RGB", img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[3])
                to_save = background
            else:
                to_save = img.convert("RGB")
            to_save.save(buf, format="JPEG", quality=quality)
            mime = "image/jpeg"
        else:
            # 默认 PNG
            to_save = img
            to_save.save(buf, format="PNG")
            mime = "image/png"
        b = buf.getvalue()
        encoded = base64.b64encode(b).decode("ascii")
        return f"data:{mime};base64,{encoded}"
    except Exception:
        return None


# -------------------------
# 从 temp_dir 收集模型输出的图片（保持文件格式信息）
# -------------------------
def collect_output_images_with_metadata(output_dir, max_images=64):
    """
    在 output_dir 下查找图片文件，返回列表 items，每项为 dict:
      { 'path': abs_path, 'filename': name, 'format': ext_upper (e.g., 'PNG','JPEG'), 'pil': PIL.Image or None }
    优先查找文件名或目录名中包含关键词（patch,crop,vis,...）
    """
    items = []
    if not output_dir or not os.path.exists(output_dir):
        return items

    exts = ('.png', '.jpg', '.jpeg', '.webp')
    keywords = ['patch', 'patches', 'crop', 'crops', 'fragment', 'frag', 'patch_img', 'crop_img', 'vis', 'visual']

    found = []
    for root, dirs, files in os.walk(output_dir):
        dir_name = os.path.basename(root).lower()
        dir_priority = any(k in dir_name for k in keywords)
        for f in files:
            if f.lower().endswith(exts):
                full = os.path.join(root, f)
                name = f.lower()
                # prefer ones with keywords
                if dir_priority or any(k in name for k in keywords):
                    found.insert(0, full)
                else:
                    found.append(full)

    # deduplicate while preserving order
    unique = []
    seen = set()
    for p in found:
        if p not in seen:
            unique.append(p)
            seen.add(p)

    for p in unique[:max_images]:
        item = {'path': p, 'filename': os.path.basename(p)}
        ext = os.path.splitext(p)[1].lower().lstrip('.')
        if ext == 'jpg':
            ext = 'jpeg'
        item['format'] = ext.upper()
        # try load PIL image (may fail for some corrupted files)
        try:
            item['pil'] = Image.open(p).convert("RGB")
        except Exception:
            item['pil'] = None
        items.append(item)

    return items


# -------------------------
# LaTeX -> 可读文本（如可用）
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
# 主要流程：运行模型推理并返回 OCR 文本 与 temp_dir 中的图片 metadata 列表
# -------------------------
def process_image_return_output_images(image, prompt_type, custom_prompt, model_size):
    """
    运行 model.infer 并在临时 output_path 中收集模型写出的图片（patches 等）。
    返回：
      - readable_text: 可读的 OCR 文本（LaTeX 转换后，如可用）
      - items: list of dict {path, filename, format, pil}
    注意：items 中的 pil 已转换为 RGB（或为 None），因此可直接传给 gr.Gallery（type='pil'）或用于 base64 编码。
    """
    try:
        model, tokenizer = load_model()
        temp_dir = tempfile.mkdtemp()

        # 保存上传的输入图片到 temp_dir（某些模型可能需要）
        temp_image_path = os.path.join(temp_dir, "input_image.jpg")
        if isinstance(image, str):
            shutil.copy(image, temp_image_path)
        else:
            image.save(temp_image_path)

        # 构建 prompt
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

        # 捕获 stdout 并运行推理（model.infer 通常会把结果写入 output_path）
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

        # 优先读取 temp_dir 中的 txt 文件（模型可能将 OCR 文本保存为 txt）
        ocr_text = ""
        for filename in os.listdir(temp_dir):
            if filename.endswith('.txt'):
                try:
                    with open(os.path.join(temp_dir, filename), 'r', encoding='utf-8') as f:
                        ocr_text += f.read() + "\n"
                except Exception:
                    pass

        # 若无 txt，则解析 captured_text（兼容老实现输出）
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

        # 在 temp_dir 中收集图片文件及其 metadata（优先带关键词的）
        items = collect_output_images_with_metadata(temp_dir, max_images=64)

        # 把临时目录内容读取到内存（PIL already loaded where possible），然后删除 temp_dir
        # Note: items[i]['pil'] may be None if loading failed; that's okay
        try:
            shutil.rmtree(temp_dir)
        except Exception:
            pass

        return readable if readable.strip() else "No text detected in image.", items

    except Exception as e:
        import traceback
        msg = f"Error: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        return msg, []


# -------------------------
# 构建 Markdown：把 items 列表（来自 temp_dir）按原始格式嵌入为 base64 data URI
# -------------------------
def build_markdown_from_items(readable_text: str, items, embed_base64=True, preserve_size=False, max_width=800, patch_max_width=600, quality=85):
    """
    items: list of dict {'path','filename','format','pil'}
    embed_base64 True: 在 md 中嵌入 data URI；否则引用相对文件名（并假设导出函数会把文件保存到同目录）
    preserve_size True: 在嵌入时尽量不缩放（使用原始文件 bytes via file_to_data_uri or pil_to_data_uri_preserve_format)
    max_width/patch_max_width: 缩放参数（当 preserve_size False 时生效）
    """
    md_lines = []
    md_lines.append("# OCR 结果\n")
    md_lines.append("## 文本\n")
    md_lines.append(readable_text or "*未识别到文本*")
    md_lines.append("\n---\n")
    md_lines.append("## 识别为图片的内容（来自模型输出目录的图片）\n")

    num = len(items) if items else 0
    md_lines.append(f"_共发现 {num} 张图片_\n\n")

    if num == 0:
        md_lines.append("_无图片_\n")
        return "\n".join(md_lines)

    for idx, it in enumerate(items, start=1):
        md_lines.append(f"### 图片片段 {idx} - `{it.get('filename','')}`\n")
        if embed_base64:
            # 如果 preserve_size 并且磁盘 path 可用，优先用 file raw bytes to keep exact format/size
            data_uri = None
            if preserve_size and it.get('path'):
                data_uri = file_to_data_uri(it['path'])
            # 否则，如果 pil 已有，按 preserve_size 决定是否缩放/原样编码
            if data_uri is None and it.get('pil') is not None:
                if preserve_size:
                    data_uri = pil_to_data_uri_preserve_format(it['pil'], original_format=it.get('format'), preserve_size=True)
                else:
                    data_uri = pil_to_data_uri_preserve_format(it['pil'], original_format=it.get('format'), preserve_size=False, max_width=patch_max_width, quality=quality)
            # 最后回退：若 path 可用，用 file raw bytes
            if data_uri is None and it.get('path'):
                data_uri = file_to_data_uri(it['path'])
            if data_uri:
                md_lines.append(f"![patch_{idx}]({data_uri})\n\n")
            else:
                # 失败则给占位
                md_lines.append(f"![patch_{idx}]()\n\n")
        else:
            # 不嵌入：使用相对文件名（导出函数会保存）
            md_lines.append(f"![patch_{idx}]({it.get('filename','patch_%d.png' % idx)})\n\n")

    return "\n".join(md_lines)


# -------------------------
# 导出 Markdown：当 embed=False 时保存原始文件到导出目录；当 embed=True 并 preserve_size=False 时 MD 已包含 resized data URI
# -------------------------
def export_markdown_items(markdown_text: str, items, embed_base64=True, preserve_size=False):
    """
    写入临时目录并返回 md 文件路径以供下载。
    如果 embed_base64 is False, 则会把 items 的文件（或 PIL）保存为相对文件（filename）到 temp dir。
    """
    try:
        temp_dir = tempfile.mkdtemp()
        md_path = os.path.join(temp_dir, "result.md")

        if not embed_base64:
            # 保存每个 item 到 temp_dir，以其原始 filename（如果不存在，则命名 patch_{i}.png）
            for idx, it in enumerate(items, start=1):
                out_name = it.get('filename') or f"patch_{idx}.png"
                out_path = os.path.join(temp_dir, out_name)
                # 优先复制原始文件 if exists
                if it.get('path') and os.path.exists(it['path']):
                    try:
                        shutil.copy(it['path'], out_path)
                        continue
                    except Exception:
                        pass
                # 否则，保存 PIL if available
                pil = it.get('pil')
                if pil is not None:
                    try:
                        pil.save(out_path, format="PNG")
                        continue
                    except Exception:
                        pass
                # else skip
        # 写 md
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(markdown_text or "")

        return md_path
    except Exception as e:
        print(f"导出失败: {e}")
        return None


# -------------------------
# Gradio 界面：左侧输入/设置，右侧结果（文本、画廊、markdown）
# -------------------------
def create_demo():
    with gr.Blocks(title="DeepSeek-OCR - Embed output images into Markdown", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🔍 DeepSeek-OCR\n将模型在临时输出目录生成的图片（patches 等）以 base64 嵌入 Markdown，并在页面上显示这些图片。")

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
                output_text = gr.Textbox(label="Extracted Text (readable)", lines=18, max_lines=300, show_copy_button=True)
                # Gallery 展示 PIL 图像列表（一些 items 可能没有 pil，Gallery 会忽略 None）
                patches_gallery = gr.Gallery(label="模型输出的图片（patches）", columns=6, type="pil")

                gr.Markdown("### 📝 Markdown / Export")
                embed_toggle = gr.Checkbox(label="在 Markdown 中嵌入图片（Base64）", value=True)
                preserve_size_checkbox = gr.Checkbox(label="尽量保持图片原始格式与尺寸（可能导致文件较大）", value=True)
                generate_md_btn = gr.Button("📝 生成 Markdown")
                md_preview = gr.Markdown(label="Markdown 预览", value="")
                export_md_btn = gr.Button("💾 导出 Markdown (.md)")
                md_file = gr.File(label="下载生成的 Markdown 文件", interactive=False)

        # Process -> 返回 OCR 文本 与 items 列表（items 会被转换为 Gallery 需要的格式：list of PIL）
        def on_process(image, prompt_type, custom_prompt, model_size):
            text, items = process_image_return_output_images(image, prompt_type, custom_prompt, model_size)
            # prepare gallery images list from items: use pil if available; otherwise skip
            gallery_imgs = []
            for it in items:
                if it.get('pil') is not None:
                    gallery_imgs.append(it['pil'])
            return text, gallery_imgs, items  # note: we'll use items separately in md generation/export

        # Gradio 不支持一个按钮同时返回三个不同类型直接映射到三个控件，
        # 我们先把 process_btn 绑定到返回文本与 Gallery，然后把 items 保存在一个隐藏组件（State）以备后续生成/导出使用。
        # 使用 gr.State 存储 items
        items_state = gr.State([])

        def process_and_store(image, prompt_type, custom_prompt, model_size):
            text, items = process_image_return_output_images(image, prompt_type, custom_prompt, model_size)
            # items 为 list of dict (path, filename, format, pil)
            # prepare gallery list
            gallery_imgs = [it['pil'] for it in items if it.get('pil') is not None]
            return text, gallery_imgs, items

        process_btn.click(
            fn=process_and_store,
            inputs=[image_input, prompt_type, custom_prompt, model_size],
            outputs=[output_text, patches_gallery, items_state]
        )

        # 生成 Markdown 回调：读取 items_state（包含 path/format/pil）并生成 md 字符串
        def on_generate_md(readable_text, items, embed_base64, preserve_size):
            # items 是 list of dicts
            md = build_markdown_from_items(readable_text, items or [], embed_base64=embed_base64, preserve_size=preserve_size, max_width=800, patch_max_width=600, quality=85)
            return md

        generate_md_btn.click(
            fn=on_generate_md,
            inputs=[output_text, items_state, embed_toggle, preserve_size_checkbox],
            outputs=[md_preview]
        )

        # 导出 md：写入临时目录并返回 md 文件路径供下载
        def on_export_md(md_str, items, embed_base64, preserve_size):
            md_path = export_markdown_items(md_str, items or [], embed_base64=embed_base64, preserve_size=preserve_size)
            return md_path

        export_md_btn.click(
            fn=on_export_md,
            inputs=[md_preview, items_state, embed_toggle, preserve_size_checkbox],
            outputs=[md_file]
        )

    return demo


if __name__ == "__main__":
    demo = create_demo()
    demo.launch(server_name="0.0.0.0", server_port=2714, share=False)
