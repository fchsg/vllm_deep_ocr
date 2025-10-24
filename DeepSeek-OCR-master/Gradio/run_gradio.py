#!/usr/bin/env python3
# run_gradio.py
# 依赖：
# pip install gradio transformers torch pillow pdf2image pylatexenc
# 系统需安装 poppler（pdf2image 依赖）。详见上方说明。

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
from pdf2image import convert_from_bytes

# LaTeX -> 可读文本
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
# 辅助函数：文件/图片 -> data URI（尽量保留原始格式）
# -------------------------
def file_to_data_uri(file_path):
    """从磁盘读取文件 bytes 并返回 data URI（保持 mime type）。"""
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
# 从模型 output 目录收集图片（并保留 path/filename/format/pil）
# -------------------------
def collect_output_images_with_metadata(output_dir, max_images=64):
    """
    在 output_dir 下查找图片文件，返回 list of dict:
      { 'path': abs_path, 'filename': name, 'format': ext_upper, 'pil': PIL.Image or None }
    优先带关键词的文件（patch/crop/vis）
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
# 单张图片推理并收集模型在 temp_dir 输出（返回 可读文本, items）
# -------------------------
def infer_image_and_collect(image_pil: Image.Image, prompt: str, config):
    """
    对单张 PIL.Image 执行 model.infer（将这张图片保存为临时文件并指定 output_path=temp_dir），
    然后从 temp_dir 中收集文本文件与图片 metadata，最终删除 temp_dir 返回结果。
    返回 (readable_text, items_list)
    """
    try:
        model, tokenizer = load_model()
        temp_dir = tempfile.mkdtemp()

        # 保存输入图像为文件供模型读取
        input_path = os.path.join(temp_dir, "input_image.jpg")
        image_pil.save(input_path)

        # 捕获 stdout（模型可能把输出打印到 stdout）
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

        # 优先读取 temp_dir 下的 txt 文件
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

        # 清理 temp_dir（items 已加载到内存的 PIL）
        try:
            shutil.rmtree(temp_dir)
        except Exception:
            pass

        return readable if readable.strip() else "No text detected in image.", items
    except Exception as e:
        import traceback
        return f"Error: {e}\n\nTraceback:\n{traceback.format_exc()}", []


# -------------------------
# 将 PDF bytes 转为 PIL.Image 分页（使用 pdf2image）
# -------------------------
def pdf_bytes_to_pil_pages(pdf_bytes, dpi=200, poppler_path=None):
    """
    把 PDF 二进制转换为每页的 PIL.Image 列表（RGB）。
    - dpi: 渲染分辨率，数值越高越清晰但越慢/大
    - poppler_path: 如果 Windows 或 poppler 未在 PATH，可传入 poppler 的 bin 路径
    """
    pages = []
    try:
        pil_pages = convert_from_bytes(pdf_bytes, dpi=dpi, poppler_path=poppler_path)
        for p in pil_pages:
            pages.append(p.convert("RGB"))
    except Exception as e:
        print("pdf2image 转换失败:", e)
    return pages


# -------------------------
# 处理整个 PDF：对每页调用 infer_image_and_collect，聚合每页结果
# -------------------------
def process_pdf_and_collect(pdf_file_path_or_obj, prompt_type, custom_prompt, model_size, dpi=200, poppler_path=None):
    """
    输入：pdf_file_path_or_obj（可能是上传的文件路径或 file-like object）
    返回：
      - aggregated_text: 按页合并的文本（每页以 '## Page N' 分隔）
      - all_items: list of items for all pages （按发现顺序）
      - per_page_items: list of lists (每页对应的 items)
    """
    # 读取 pdf bytes
    try:
        if isinstance(pdf_file_path_or_obj, str) and os.path.exists(pdf_file_path_or_obj):
            with open(pdf_file_path_or_obj, "rb") as f:
                pdf_bytes = f.read()
        else:
            # gr.File 传给我们的可能是一个临时文件对象
            pdf_bytes = pdf_file_path_or_obj.read()
    except Exception as e:
        return f"无法读取上传的 PDF: {e}", [], []

    pages = pdf_bytes_to_pil_pages(pdf_bytes, dpi=dpi, poppler_path=poppler_path)
    if not pages:
        return "无法将 PDF 转为图片或 PDF 为空。", [], []

    # 模型 size config map（与单图推理使用的配置一致）
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

    # 为每页设置 prompt（可自定义）
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
        # extend overall items
        all_items.extend(items)

    aggregated_text = "\n\n".join(aggregated_text_parts)
    return aggregated_text, all_items, per_page_items


# -------------------------
# 构建 Markdown：把 items 列表以 base64 嵌入（保持原始格式尽量）
# -------------------------
def build_markdown_from_items_grouped(aggregated_text: str, per_page_items, embed_base64=True, preserve_size=True, patch_max_width=800, quality=90):
    """
    per_page_items: list of lists (each inner list items for that page)
    embed_base64: 是否在 md 中嵌入 data URI（True 推荐）
    preserve_size: True 尽量使用原始 bytes（file_to_data_uri），否则进行缩放编码
    返回 markdown 字符串
    """
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
                # 优先使用原始文件 bytes（如果临时目录尚存），但在我们的流程中 temp_dir 已被删除，
                # 所以优先使用 PIL 并 preserve_size 决定是否缩放或选择 PNG/JPEG
                pil = it.get('pil')
                if pil is not None:
                    if preserve_size:
                        data_uri = pil_to_data_uri(pil, original_format=it.get('format'), preserve_size=True)
                    else:
                        data_uri = pil_to_data_uri(pil, original_format=it.get('format'), preserve_size=False, max_width=patch_max_width, quality=quality)
                # 最后回退：如 path 仍可访问，则用 file_to_data_uri
                if data_uri is None and it.get('path'):
                    data_uri = file_to_data_uri(it['path'])
                if data_uri:
                    md_lines.append(f"![page{page_idx}_patch{idx}]({data_uri})\n")
                else:
                    md_lines.append("![missing]()\n")
            else:
                # 不嵌入：引用文件名，导出函数会把相应文件写入导出目录
                md_lines.append(f"![{filename}]({filename})\n")
        md_lines.append("\n")
    return "\n".join(md_lines)


# -------------------------
# 导出 Markdown：写入临时目录并返回 md 路径（当 embed=False 时也会把图片保存）
# -------------------------
def export_markdown_pdf(md_text: str, per_page_items, embed_base64=True):
    """
    若 embed_base64 False，会把 per_page_items 的原始文件或 PIL 保存到 temp_dir，
    md_text 已应使用相对文件名引用图片（build_markdown_from_items_grouped 的非嵌入分支）。
    返回 md 文件绝对路径。
    """
    try:
        temp_dir = tempfile.mkdtemp()
        # 如果需要保存图片文件
        if not embed_base64:
            for page_idx, items in enumerate(per_page_items, start=1):
                for idx, it in enumerate(items, start=1):
                    out_name = it.get('filename') or f'page{page_idx}_patch{idx}.png'
                    out_path = os.path.join(temp_dir, out_name)
                    # 先 try 原始 path
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
# Gradio 界面（包含 PDF 上传/处理）
# -------------------------
def create_demo():
    with gr.Blocks(title="DeepSeek-OCR (PDF support)", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# DeepSeek-OCR — PDF OCR 支持\n上传 PDF，逐页 OCR 并把模型输出目录的图片以 base64 嵌入 Markdown。")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 输入 / 设置")
                image_input = gr.Image(label="单图上传（若要单页测试）", type="pil", sources=["upload", "clipboard"])
                pdf_input = gr.File(label="上传 PDF（.pdf）", file_types=['.pdf'])
                prompt_type = gr.Radio(choices=["Free OCR", "Markdown Conversion", "Custom"], value="Markdown Conversion", label="Prompt Type")
                custom_prompt = gr.Textbox(label="Custom Prompt (if selected)", placeholder="Enter custom prompt...", lines=2, visible=False)
                model_size = gr.Radio(choices=["Tiny", "Small", "Base", "Large", "Gundam (Recommended)"], value="Gundam (Recommended)", label="Model Size")
                dpi_slider = gr.Slider(100, 400, value=200, step=10, label="PDF 渲染 DPI（越高越清晰，越慢/占用越大）")
                poppler_path_input = gr.Textbox(label="poppler bin 路径（Windows 可选）", placeholder="如果 poppler 未在 PATH，填写 poppler 的 bin 路径", value="", visible=False)
                process_image_btn = gr.Button("处理单张图片", variant="primary")
                process_pdf_btn = gr.Button("处理 PDF", variant="secondary")

                def update_custom_visible(choice):
                    return gr.update(visible=(choice == "Custom"))
                prompt_type.change(fn=update_custom_visible, inputs=[prompt_type], outputs=[custom_prompt])

                def update_poppler_visible(os_choice):
                    # 这里不做 OS 判断，保留控件以便用户填写
                    return gr.update(visible=True)
                # 可按需显示 poppler_path_input
                # prompt_type.change(fn=update_poppler_visible, inputs=[prompt_type], outputs=[poppler_path_input])

            with gr.Column(scale=1):
                gr.Markdown("### 结果 / Markdown")
                output_text = gr.Textbox(label="OCR 文本（单图或 PDF 合并）", lines=20, max_lines=500, show_copy_button=True)
                gallery = gr.Gallery(label="模型输出的图片（patches/crops）", columns=6, type="pil")
                embed_toggle = gr.Checkbox(label="在 Markdown 中嵌入图片（Base64）", value=True)
                preserve_size = gr.Checkbox(label="尽量保持原始格式/尺寸（可能导致文件较大）", value=True)
                generate_md_btn = gr.Button("生成 Markdown")
                md_preview = gr.Markdown(label="Markdown 预览", value="")
                export_md_btn = gr.Button("导出 Markdown (.md)")
                md_file = gr.File(label="下载生成的 Markdown 文件", interactive=False)

        # state 保存 items（per_page_items）
        per_page_items_state = gr.State([])

        # 处理单图按钮：调用 infer_image_and_collect（直接返回文本与 gallery images，并把 items 存入 state）
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
            # per_page_items_state 用于兼容 PDF 流程（单图视为一页）
            return text, gallery_imgs, [items]

        process_image_btn.click(
            fn=on_process_image,
            inputs=[image_input, prompt_type, custom_prompt, model_size],
            outputs=[output_text, gallery, per_page_items_state]
        )

        # 处理 PDF 按钮
        def on_process_pdf(pdf_file, prompt_type, custom_prompt, model_size, dpi, poppler_path):
            if pdf_file is None:
                return "请上传 PDF 文件", [], []
            text, all_items, per_page_items = process_pdf_and_collect(pdf_file, prompt_type, custom_prompt, model_size, dpi=dpi, poppler_path=poppler_path or None)
            # gallery 显示所有 items 中的 pil
            gallery_imgs = [it['pil'] for it in all_items if it.get('pil') is not None]
            return text, gallery_imgs, per_page_items

        process_pdf_btn.click(
            fn=on_process_pdf,
            inputs=[pdf_input, prompt_type, custom_prompt, model_size, dpi_slider, poppler_path_input],
            outputs=[output_text, gallery, per_page_items_state]
        )

        # 生成 Markdown（从 per_page_items_state 生成）
        def on_generate_md(aggregated_text, per_page_items, embed_base64_flag, preserve_size_flag):
            per_page_items = per_page_items or []
            md = build_markdown_from_items_grouped(aggregated_text or "", per_page_items, embed_base64=embed_base64_flag, preserve_size=preserve_size_flag)
            return md

        generate_md_btn.click(
            fn=on_generate_md,
            inputs=[output_text, per_page_items_state, embed_toggle, preserve_size],
            outputs=[md_preview]
        )

        # 导出 Markdown（写文件并返回 md 路径）
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
    demo.launch(server_name="0.0.0.0", server_port=2714, share=False)
