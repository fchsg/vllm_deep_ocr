#!/usr/bin/env python3
# run_gradio.py
# 依赖：
# pip install gradio transformers torch pillow pymupdf pylatexenc
# 如果使用 GPU，请确保 torch 已安装对应 CUDA 版本

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

# 使用 PyMuPDF（pymupdf）
import fitz  # pip install pymupdf

# LaTeX -> 可读文本（可选）
try:
    from pylatexenc.latex2text import LatexNodes2Text
except Exception:
    LatexNodes2Text = None
    print("警告: pylatexenc 未安装，LaTeX 转文本将回退到原文。可执行: pip install pylatexenc")

# 全局延迟加载模型与 tokenizer
model = None
tokenizer = None


def load_model():
    """延迟加载 DeepSeek-OCR 模型与 tokenizer。优先 GPU:bfloat16，否则回退 CPU:float32。"""
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
        try:
            model = model.eval().cuda().to(torch.bfloat16)
            print("Using CUDA + bfloat16")
        except Exception:
            model = model.eval().to(torch.float32)
            print("CUDA not available or error: using CPU + float32")
        print("Model loaded.")
    return model, tokenizer


# -------------------------
# 辅助：读取文件 bytes -> data URI（尽量保留原始格式）
# -------------------------
def file_to_data_uri(file_path):
    """ 从磁盘读取文件 bytes 并返回 data URI（保持 mime type）。 """
    try:
        with open(file_path, "rb") as f:
            b = f.read()
        mime, _ = mimetypes.guess_type(file_path)
        if not mime:
            mime = "image/jpeg"
        encoded = base64.b64encode(b).decode("ascii")
        return f"data:{mime};base64,{encoded}"
    except Exception:
        return None


def pil_to_data_uri(pil_img: Image.Image, original_format=None, preserve_size=False, max_width=None, quality=85):
    """
    将 PIL.Image 编码为 data URI。
    - original_format: 'PNG'/'JPEG' 等，若提供则尝试用该格式保存
    - preserve_size: True 不缩放；False 可根据 max_width 缩放
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
            if img.mode in ("RGBA", "LA"):
                background = Image.new("RGB", img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[3])
                to_save = background
            else:
                to_save = img.convert("RGB")
            to_save.save(buf, format="JPEG", quality=quality)
            mime = "image/jpeg"
        else:
            img.save(buf, format="PNG")
            mime = "image/png"
        b = buf.getvalue()
        encoded = base64.b64encode(b).decode("ascii")
        return f"data:{mime};base64,{encoded}"
    except Exception:
        return None


# -------------------------
# 在模型 output 目录中收集图片（保留 path/filename/format/pil）
# -------------------------
def collect_output_images_with_metadata(output_dir, max_images=64):
    """
    在 output_dir 下查找图片文件，返回 list of dict:
      { 'path': abs_path, 'filename': name, 'format': ext_upper, 'pil': PIL.Image or None }
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
                if dir_priority or any(k in name for k in keywords):
                    found.insert(0, full)
                else:
                    found.append(full)

    # 去重并读取
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
        try:
            item['pil'] = Image.open(p).convert("RGB")
        except Exception:
            item['pil'] = None
        items.append(item)
    return items


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
# 对单张 PIL 图片调用模型并收集 temp_dir 中输出
# -------------------------
def infer_image_and_collect(image_pil: Image.Image, prompt: str, config):
    """
    对单张 PIL.Image 执行 model.infer，并从临时 output_path 收集文本和图片 metadata。
    返回 (readable_text, items_list)
    """
    try:
        model, tokenizer = load_model()
        temp_dir = tempfile.mkdtemp()

        input_path = os.path.join(temp_dir, "input_image.jpg")
        image_pil.save(input_path)

        # 捕获 stdout
        import sys
        from io import StringIO
        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()
        try:
            result = model.infer(
                tokenizer,
                prompt=prompt,
                image_file=input_path,
                output_path=temp_dir,
                base_size=config.get("base_size", 1024),
                image_size=config.get("image_size", 1024),
                crop_mode=config.get("crop_mode", False),
                save_results=True,
                test_compress=False
            )
        finally:
            sys.stdout = old_stdout

        captured_text = captured_output.getvalue()

        # 读取 temp_dir 中的 txt 文件为 OCR 文本
        ocr_text = ""
        for fname in os.listdir(temp_dir):
            if fname.endswith('.txt'):
                try:
                    with open(os.path.join(temp_dir, fname), 'r', encoding='utf-8') as f:
                        ocr_text += f.read() + "\n"
                except Exception:
                    pass

        if not ocr_text.strip() and captured_text.strip():
            lines = captured_text.splitlines()
            clean = []
            for line in lines:
                if '<|ref|>' in line or '<|det|>' in line or '<|/ref|>' in line or '<|/det|>' in line:
                    m = re.search(r'<\|/ref\|>(.*?)<\|det\|>', line)
                    if m:
                        clean.append(m.group(1).strip())
                elif line.startswith('=====') or 'BASE:' in line or 'PATCHES:' in line or line.startswith('image:') or line.startswith('other:'):
                    continue
                elif line.strip():
                    clean.append(line.strip())
            ocr_text = "\n".join(clean)

        if not ocr_text.strip() and isinstance(result, str):
            ocr_text = result

        readable = latex_to_readable_text(ocr_text) if ocr_text else ""

        # 收集 temp_dir 中模型生成的图片
        items = collect_output_images_with_metadata(temp_dir, max_images=64)

        # 清理临时目录（items 已加载到内存）
        try:
            shutil.rmtree(temp_dir)
        except Exception:
            pass

        return readable if readable.strip() else "No text detected in image.", items
    except Exception as e:
        import traceback
        return f"Error: {e}\n\nTraceback:\n{traceback.format_exc()}", []


# -------------------------
# 使用 PyMuPDF 把 PDF bytes 转为 PIL.Image 页列表
# -------------------------
def pdf_bytes_to_pil_pages_with_pymupdf(pdf_bytes, zoom=2.0):
    """
    将 PDF bytes 转为每页 PIL.Image（RGB）。
    - zoom: 渲染缩放因子，approx DPI = 72 * zoom。默认 2.0 ~ 144 DPI。
    """
    pages = []
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        for page in doc:
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            pages.append(img)
        doc.close()
    except Exception as e:
        print("PyMuPDF 转换 PDF->image 失败:", e)
    return pages


# -------------------------
# 处理整个 PDF：对每页调用 infer_image_and_collect，聚合结果
# -------------------------
def process_pdf_and_collect(pdf_file_path_or_obj, prompt_type, custom_prompt, model_size, zoom=2.0):
    """
    pdf_file_path_or_obj: 上传的文件路径或 file-like 对象
    返回 aggregated_text, all_items, per_page_items
    """
    # 读取 pdf bytes
    try:
        if isinstance(pdf_file_path_or_obj, str) and os.path.exists(pdf_file_path_or_obj):
            with open(pdf_file_path_or_obj, "rb") as f:
                pdf_bytes = f.read()
        else:
            pdf_bytes = pdf_file_path_or_obj.read()
    except Exception as e:
        return f"无法读取上传的 PDF: {e}", [], []

    pages = pdf_bytes_to_pil_pages_with_pymupdf(pdf_bytes, zoom=zoom)
    if not pages:
        return "无法将 PDF 转为图片或 PDF 为空。", [], []

    size_map = {
        "Tiny": {"base_size": 512, "image_size": 512, "crop_mode": False},
        "Small": {"base_size": 640, "image_size": 640, "crop_mode": False},
        "Base": {"base_size": 1024, "image_size": 1024, "crop_mode": False},
        "Large": {"base_size": 1280, "image_size": 1280, "crop_mode": False},
        "Gundam (Recommended)": {"base_size": 1024, "image_size": 640, "crop_mode": True}
    }
    config = size_map.get(model_size, size_map["Gundam (Recommended)"])

    aggregated_text_parts = []
    all_items = []
    per_page_items = []

    for idx, page_img in enumerate(pages, start=1):
        if prompt_type == "Free OCR":
            prompt = "<image>\nFree OCR. "
        elif prompt_type == "Markdown Conversion":
            prompt = "<image>\n<|grounding|>Convert the document to markdown. "
        elif prompt_type == "Custom":
            prompt = f"<image>\n{custom_prompt}"
        else:
            prompt = "<image>\nFree OCR. "

        text, items = infer_image_and_collect(page_img, prompt, config)
        aggregated_text_parts.append(f"## Page {idx}\n\n{text}")
        per_page_items.append(items)
        all_items.extend(items)

    aggregated_text = "\n\n".join(aggregated_text_parts)
    return aggregated_text, all_items, per_page_items


# -------------------------
# 构建 Markdown：按页将 items 嵌入为 data URI（尽量保持原始格式）
# -------------------------
def build_markdown_from_items_grouped(aggregated_text: str, per_page_items, embed_base64=True, preserve_size=True, patch_max_width=800, quality=90):
    md_lines = []
    md_lines.append("# OCR 结果（PDF）\n")
    md_lines.append("## 文本（按页）\n")
    md_lines.append(aggregated_text or "*未识别到文本*")
    md_lines.append("\n---\n")
    md_lines.append("## 模型输出的图片（按页组织）\n")

    if not per_page_items:
        md_lines.append("_无图片_\n")
        return "\n".join(md_lines)

    for page_idx, items in enumerate(per_page_items, start=1):
        md_lines.append(f"### Page {page_idx} 图片片段（共 {len(items)} 张）\n")
        if not items:
            md_lines.append("_无图片_\n")
            continue
        for idx, it in enumerate(items, start=1):
            filename = it.get('filename', f'page{page_idx}_patch{idx}')
            md_lines.append(f"#### Page {page_idx} - 片段 {idx} - `{filename}`\n")
            if embed_base64:
                data_uri = None
                pil = it.get('pil')
                if pil is not None:
                    if preserve_size:
                        data_uri = pil_to_data_uri(pil, original_format=it.get('format'), preserve_size=True)
                    else:
                        data_uri = pil_to_data_uri(pil, original_format=it.get('format'), preserve_size=False, max_width=patch_max_width, quality=quality)
                if data_uri is None and it.get('path'):
                    data_uri = file_to_data_uri(it['path'])
                if data_uri:
                    md_lines.append(f"![page{page_idx}_patch{idx}]({data_uri})\n")
                else:
                    md_lines.append("![missing]()\n")
            else:
                md_lines.append(f"![{filename}]({filename})\n")
        md_lines.append("\n")
    return "\n".join(md_lines)


# -------------------------
# 导出 Markdown：将 md 写入临时目录并返回路径（若不嵌入则保存图片文件）
# -------------------------
def export_markdown_pdf(md_text: str, per_page_items, embed_base64=True):
    try:
        temp_dir = tempfile.mkdtemp()
        if not embed_base64:
            for page_idx, items in enumerate(per_page_items, start=1):
                for idx, it in enumerate(items, start=1):
                    out_name = it.get('filename') or f'page{page_idx}_patch{idx}.png'
                    out_path = os.path.join(temp_dir, out_name)
                    if it.get('path') and os.path.exists(it['path']):
                        try:
                            shutil.copy(it['path'], out_path)
                            continue
                        except Exception:
                            pass
                    pil = it.get('pil')
                    if pil is not None:
                        try:
                            pil.save(out_path, format="PNG")
                        except Exception:
                            try:
                                pil.save(out_path.replace('.png', '.jpg'), format="JPEG", quality=90)
                            except Exception:
                                pass
        md_path = os.path.join(temp_dir, "result.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_text or "")
        return md_path
    except Exception as e:
        print("导出失败:", e)
        return None


# -------------------------
# Gradio 界面（含 PDF 上传/处理）
# -------------------------
def create_demo():
    with gr.Blocks(title="DeepSeek-OCR (PDF via PyMuPDF)", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# DeepSeek-OCR — PDF OCR 支持（使用 PyMuPDF）\n上传 PDF，逐页 OCR 并把模型输出的图片以 base64 嵌入 Markdown。")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 输入 / 设置")
                image_input = gr.Image(label="单图上传（若要单页测试）", type="pil", sources=["upload", "clipboard"])
                pdf_input = gr.File(label="上传 PDF（.pdf）", file_types=['.pdf'])
                prompt_type = gr.Radio(choices=["Free OCR", "Markdown Conversion", "Custom"], value="Markdown Conversion", label="Prompt Type")
                custom_prompt = gr.Textbox(label="Custom Prompt (if selected)", placeholder="Enter custom prompt...", lines=2, visible=False)
                model_size = gr.Radio(choices=["Tiny", "Small", "Base", "Large", "Gundam (Recommended)"], value="Gundam (Recommended)", label="Model Size")
                zoom_slider = gr.Slider(1.0, 4.0, value=2.0, step=0.1, label="PDF 渲染 zoom（大致相当于 DPI/72）")
                process_image_btn = gr.Button("处理单张图片", variant="primary")
                process_pdf_btn = gr.Button("处理 PDF", variant="secondary")

                def update_custom_visible(choice):
                    return gr.update(visible=(choice == "Custom"))
                prompt_type.change(fn=update_custom_visible, inputs=[prompt_type], outputs=[custom_prompt])

            with gr.Column(scale=1):
                gr.Markdown("### 结果 / Markdown")
                output_text = gr.Textbox(label="OCR 文本（单图或 PDF 合并）", lines=20, max_lines=500, show_copy_button=True)
                gallery = gr.Gallery(label="模型输出的图片（patches/crops）", columns=6, type="pil")
                embed_toggle = gr.Checkbox(label="在 Markdown 中嵌入图片（Base64）", value=True)
                preserve_size = gr.Checkbox(label="尽量保持原始格式/尺寸（可能导致文件很大）", value=True)
                generate_md_btn = gr.Button("生成 Markdown")
                md_preview = gr.Markdown(label="Markdown 预览", value="")
                export_md_btn = gr.Button("导出 Markdown (.md)")
                md_file = gr.File(label="下载生成的 Markdown 文件", interactive=False)

        per_page_items_state = gr.State([])

        # 单图处理
        def on_process_image(image, prompt_type, custom_prompt, model_size):
            if image is None:
                return "请上传图片", [], []
            if prompt_type == "Free OCR":
                prompt = "<image>\nFree OCR. "
            elif prompt_type == "Markdown Conversion":
                prompt = "<image>\n<|grounding|>Convert the document to markdown. "
            else:
                prompt = f"<image>\n{custom_prompt}"
            size_map = {
                "Tiny": {"base_size": 512, "image_size": 512, "crop_mode": False},
                "Small": {"base_size": 640, "image_size": 640, "crop_mode": False},
                "Base": {"base_size": 1024, "image_size": 1024, "crop_mode": False},
                "Large": {"base_size": 1280, "image_size": 1280, "crop_mode": False},
                "Gundam (Recommended)": {"base_size": 1024, "image_size": 640, "crop_mode": True}
            }
            config = size_map.get(model_size, size_map["Gundam (Recommended)"])
            text, items = infer_image_and_collect(image, prompt, config)
            gallery_imgs = [it['pil'] for it in items if it.get('pil') is not None]
            return text, gallery_imgs, [items]

        process_image_btn.click(
            fn=on_process_image,
            inputs=[image_input, prompt_type, custom_prompt, model_size],
            outputs=[output_text, gallery, per_page_items_state]
        )

        # 处理 PDF
        def on_process_pdf(pdf_file, prompt_type, custom_prompt, model_size, zoom):
            if pdf_file is None:
                return "请上传 PDF 文件", [], []
            text, all_items, per_page_items = process_pdf_and_collect(pdf_file, prompt_type, custom_prompt, model_size, zoom=zoom)
            gallery_imgs = [it['pil'] for it in all_items if it.get('pil') is not None]
            return text, gallery_imgs, per_page_items

        process_pdf_btn.click(
            fn=on_process_pdf,
            inputs=[pdf_input, prompt_type, custom_prompt, model_size, zoom_slider],
            outputs=[output_text, gallery, per_page_items_state]
        )

        # 生成 Markdown
        def on_generate_md(aggregated_text, per_page_items, embed_base64_flag, preserve_size_flag):
            per_page_items = per_page_items or []
            md = build_markdown_from_items_grouped(aggregated_text or "", per_page_items, embed_base64=embed_base64_flag, preserve_size=preserve_size_flag)
            return md

        generate_md_btn.click(
            fn=on_generate_md,
            inputs=[output_text, per_page_items_state, embed_toggle, preserve_size],
            outputs=[md_preview]
        )

        # 导出 Markdown
        def on_export_md(md_str, per_page_items, embed_base64_flag, preserve_size_flag):
            per_page_items = per_page_items or []
            md_path = export_markdown_pdf(md_str, per_page_items, embed_base64=embed_base64_flag)
            return md_path

        export_md_btn.click(
            fn=on_export_md,
            inputs=[md_preview, per_page_items_state, embed_toggle, preserve_size],
            outputs=[md_file]
        )

    return demo


if __name__ == "__main__":
    demo = create_demo()
    demo.launch(server_name="0.0.0.0", server_port=7214, share=False)
