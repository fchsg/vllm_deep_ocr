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

# pylatexenc 用于 LaTeX -> 可读文本
try:
    from pylatexenc.latex2text import LatexNodes2Text
except Exception:
    LatexNodes2Text = None
    print("警告: pylatexenc 未安装，LaTeX 转文本将回退到原文。可执行: pip install pylatexenc")

# 全局模型变量
model = None
tokenizer = None


def load_model():
    """延迟加载 DeepSeek-OCR 模型与 tokenizer"""
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
        # 默认尝试 GPU + bfloat16；如需改为 CPU，请修改下面一行
        try:
            model = model.eval().cuda().to(torch.bfloat16)
        except Exception:
            # 如果没有 GPU，则退回 CPU float32
            model = model.eval().to(torch.float32)
        print("Model loaded successfully!")
    return model, tokenizer


# -------------------------
# PIL -> base64 data URI 辅助
# -------------------------
def pil_image_to_base64_datauri(img: Image.Image, max_width=800, quality=85, fmt="JPEG"):
    """将 PIL.Image 转为 base64 data URI（压缩/缩放以控制大小）"""
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


# -------------------------
# 收集模型输出中的碎图（patches）
# -------------------------
def collect_patch_images(output_dir, max_patches=24):
    """
    在模型输出目录中收集可能的碎图（patches / crops / fragments）。
    返回 PIL.Image 列表（convert("RGB")）。
    """
    if not output_dir or not os.path.exists(output_dir):
        return []

    exts = ('.png', '.jpg', '.jpeg', '.webp')
    keywords = ['patch', 'patches', 'crop', 'crops', 'fragment', 'frag', 'cropbox', 'box', 'crop_img', 'patch_img', 'vis']
    found_paths = []

    # 优先找带关键词的文件或目录
    for root, dirs, files in os.walk(output_dir):
        dir_name = os.path.basename(root).lower()
        dir_priority = any(k in dir_name for k in keywords)
        for f in files:
            if f.lower().endswith(exts):
                full = os.path.join(root, f)
                fname = f.lower()
                if dir_priority or any(k in fname for k in keywords):
                    found_paths.append(full)

    # 回退查找任意图片
    if not found_paths:
        for root, dirs, files in os.walk(output_dir):
            for f in files:
                if f.lower().endswith(exts):
                    found_paths.append(os.path.join(root, f))

    # 去重并按发现顺序
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
# 解析模型输出文本 -> OCR 文本（并尝试 latex 转换）
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
# 主流程：运行模型推理并返回 (readable_text, patches_list)
# -------------------------
def process_image_collect_patches(image, prompt_type, custom_prompt, model_size):
    """
    运行模型推理并收集作为“图片”的识别结果（patches）。
    返回：
      - readable_text: LaTeX 转换后的可读文本（用于 Results）
      - patches: list[PIL.Image]（用于 Gallery 和 Markdown 嵌入）
    """
    try:
        model, tokenizer = load_model()
        temp_dir = tempfile.mkdtemp()

        # 保存上传图像到临时目录
        temp_image_path = os.path.join(temp_dir, "input_image.jpg")
        if isinstance(image, str):
            shutil.copy(image, temp_image_path)
        else:
            image.save(temp_image_path)

        # prompt 构建
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

        # 捕获 stdout 并执行模型推理（模型可能会将文本/patches写入 temp_dir）
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

        # 1) 优先读取 temp_dir 中的文本文件
        ocr_text = ""
        for filename in os.listdir(temp_dir):
            if filename.endswith('.txt'):
                try:
                    with open(os.path.join(temp_dir, filename), 'r', encoding='utf-8') as f:
                        ocr_text += f.read() + "\n"
                except Exception:
                    pass

        # 2) 如果没有 txt，则从 captured_text 解析
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

        # 3) fallback 如果 result 是字符串
        if not ocr_text.strip():
            if isinstance(result, str):
                ocr_text = result
            else:
                ocr_text = ""

        # 转换 LaTeX 到可读文本（如果可用）
        readable = latex_to_readable_text(ocr_text) if ocr_text else ""

        # 收集 patches（模型输出目录中的小图片）
        patches = collect_patch_images(temp_dir, max_patches=32)

        # 如果模型没有输出碎图，但 OCR 文本中可能包含 bbox 信息，可以在此解析并裁剪原图生成 patches
        # （如需解析 bbox 并裁剪，请提供 captured_text 的示例格式，我可以帮你添加裁剪逻辑）

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
# 将 patches 列表转换为 base64 列表
# -------------------------
def images_to_base64_list(images, max_width=600, quality=85, fmt="JPEG"):
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
# 构建 Markdown（把 patches 嵌入为 base64 并按样式组织）
# -------------------------
def build_markdown_from_text_and_patches(readable_text: str, patches_images=None, patches_base64=None,
                                        embed_base64=True, max_width=1200, patch_max_width=600, quality=85):
    """
    readable_text: OCR 可读文本
    patches_images: list[PIL.Image]
    patches_base64: list[data uri strings]
    embed_base64: 是否在 md 中嵌入 base64（True 推荐）
    返回 markdown 字符串
    """
    if patches_base64 is None and patches_images:
        patches_base64 = images_to_base64_list(patches_images, max_width=patch_max_width, quality=quality)

    md = []
    md.append("# OCR 结果\n")
    md.append("## 文本\n")
    md.append(readable_text or "*未识别到文本*")
    md.append("\n---\n")
    md.append("## 识别为图片的内容（碎图）\n")
    num = len(patches_base64) if patches_base64 else (len(patches_images) if patches_images else 0)
    md.append(f"_共识别到 {num} 张图片样式的碎图_\n\n")

    if num == 0:
        md.append("_无识别图片_\n")
    else:
        for idx, uri in enumerate(patches_base64 or [], start=1):
            md.append(f"### 图片片段 {idx}\n")
            md.append(f"![patch_{idx}]({uri})\n")
            md.append("\n")
    return "\n".join(md)


# -------------------------
# 导出 Markdown（嵌入或不嵌入）
# -------------------------
def export_markdown_with_patches(markdown_text: str, patches_images, embed_base64=True, quality=85):
    """
    将 markdown 写到临时目录并返回文件路径用于 gr.File 下载。
    如果 embed_base64=False，会把 patches_images 单独保存到目录并在 md 中使用相对路径（但这里我们默认 embed=True）
    """
    try:
        temp_dir = tempfile.mkdtemp()
        md_path = os.path.join(temp_dir, "result.md")

        if not embed_base64:
            # 如果不嵌入，则需要保存 patches 并替换 md 中的引用 — 当前实现以 embed=True 为主，用不到此路径
            for idx, p in enumerate(patches_images or [], start=1):
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
# Gradio 界面（左侧输入，右侧结果）
# -------------------------
def create_demo():
    with gr.Blocks(title="DeepSeek-OCR (patches as images)", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🔍 DeepSeek-OCR - 将识别为图片的内容嵌入 Markdown\n上传图片 -> OCR -> 返回可读文本与识别为图片的碎图（patches）。生成 Markdown 时以 base64 嵌入这些碎图。")

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

                gr.Markdown("### 📝 Markdown / Export")
                readable_toggle = gr.Checkbox(label="将 LaTeX 转换为可读文本（pylatexenc）", value=True)
                embed_toggle = gr.Checkbox(label="在 Markdown 中嵌入图片（Base64）", value=True)
                generate_md_btn = gr.Button("📝 生成 Markdown")
                md_preview = gr.Markdown(label="Markdown 预览", value="")
                export_md_btn = gr.Button("💾 导出 Markdown (.md)")
                md_file = gr.File(label="下载生成的 Markdown 文件", interactive=False)

        # Process: 得到可读文本与 patches
        process_btn.click(
            fn=process_image_collect_patches,
            inputs=[image_input, prompt_type, custom_prompt, model_size],
            outputs=[output_text, patches_gallery]
        )

        # 生成 Markdown
        def to_md(text_result, patches_list, use_readable, embed_base64):
            # text_result 在 process 已经为可读文本（如果 process 做了 latex 转换）
            readable = text_result or ""
            patches_images = patches_list or []
            md = build_markdown_from_text_and_patches(readable, patches_images=patches_images, patches_base64=None, embed_base64=embed_base64)
            return md

        generate_md_btn.click(
            fn=to_md,
            inputs=[output_text, patches_gallery, readable_toggle, embed_toggle],
            outputs=[md_preview]
        )

        # 导出 Markdown（返回 md 文件路径）
        def export_md(md_str, patches_list, embed_base64):
            patches_images = patches_list or []
            md_path = export_markdown_with_patches(md_str, patches_images, embed_base64=embed_base64)
            return md_path

        export_md_btn.click(
            fn=export_md,
            inputs=[md_preview, patches_gallery, embed_toggle],
            outputs=[md_file]
        )

    return demo


if __name__ == "__main__":
    demo = create_demo()
    demo.launch(server_name="0.0.0.0", server_port=2714, share=False)
