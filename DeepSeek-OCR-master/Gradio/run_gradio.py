# run_gradio.py
# 确保已安装依赖：pip install gradio transformers torch pillow pylatexenc

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
        # 默认尝试 GPU + bfloat16；如需改为 CPU，请修改此处
        model = model.eval().cuda().to(torch.bfloat16)
        print("Model loaded successfully!")
    return model, tokenizer


# -------------------------
# 辅助：PIL Image -> base64 data URI
# -------------------------
def pil_image_to_base64_datauri(img: Image.Image, max_width=1200, quality=85, fmt="JPEG"):
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
# 收集模型输出目录中的碎图（patches）与可视化整图
# -------------------------
def collect_patch_images(output_dir, max_patches=16):
    if not output_dir or not os.path.exists(output_dir):
        return []
    exts = ('.png', '.jpg', '.jpeg', '.webp')
    keywords = ['patch', 'patches', 'crop', 'crops', 'fragment', 'frag', 'cropbox', 'box', 'crop_img', 'patch_img', 'vis']
    found_paths = []
    for root, dirs, files in os.walk(output_dir):
        dir_name = os.path.basename(root).lower()
        dir_priority = any(k in dir_name for k in keywords)
        for f in files:
            if f.lower().endswith(exts):
                full = os.path.join(root, f)
                fname = f.lower()
                if dir_priority or any(k in fname for k in keywords):
                    found_paths.append(full)
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


def find_result_image_in_dir(dir_path):
    if not dir_path or not os.path.exists(dir_path):
        return None
    exts = ('.png', '.jpg', '.jpeg', '.webp')
    candidates = []
    for fname in os.listdir(dir_path):
        if fname.lower().endswith(exts):
            candidates.append(fname)
    if not candidates:
        return None
    preferred_keywords = ['result', 'vis', 'output', 'pred', 'ocr']
    def score(name):
        n = name.lower()
        s = 0
        for i, kw in enumerate(preferred_keywords):
            if kw in n:
                s += (len(preferred_keywords) - i) * 10
        try:
            p = os.path.getsize(os.path.join(dir_path, name))
            s += int(p / 1024)
        except Exception:
            pass
        return s
    candidates.sort(key=lambda x: score(x), reverse=True)
    best = candidates[0]
    try:
        img_path = os.path.join(dir_path, best)
        pil_img = Image.open(img_path).convert("RGB")
        return pil_img
    except Exception:
        return None


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
# OCR 处理主逻辑：返回 (readable_text, result_img, patches_list)
# -------------------------
def process_image_full(image, prompt_type, custom_prompt, model_size):
    try:
        model, tokenizer = load_model()
        temp_dir = tempfile.mkdtemp()

        # 保存上传图像
        temp_image_path = os.path.join(temp_dir, "input_image.jpg")
        if isinstance(image, str):
            shutil.copy(image, temp_image_path)
            input_pil = Image.open(temp_image_path).convert("RGB")
        else:
            image.save(temp_image_path)
            input_pil = image.convert("RGB")

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

        # 捕获 stdout 以兼容模型输出到 stdout 的实现
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

        # 优先读取 output txt 文件
        ocr_text = ""
        for filename in os.listdir(temp_dir):
            if filename.endswith('.txt'):
                try:
                    with open(os.path.join(temp_dir, filename), 'r', encoding='utf-8') as f:
                        ocr_text += f.read() + "\n"
                except Exception:
                    pass

        # 若无 txt，则解析 captured_text
        if not ocr_text.strip() and captured_text.strip():
            lines = captured_text.split('\n')
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

        if not ocr_text.strip():
            if isinstance(result, str):
                ocr_text = result
            else:
                ocr_text = ""

        readable = latex_to_readable_text(ocr_text) if ocr_text else ""

        # 查找模型输出整图与碎图
        result_img = find_result_image_in_dir(temp_dir)
        if result_img is None:
            result_img = input_pil

        patches = collect_patch_images(temp_dir, max_patches=32)

        # 清理临时目录（图片已加载到内存）
        try:
            shutil.rmtree(temp_dir)
        except Exception:
            pass

        return readable if readable.strip() else "No text detected in image.", result_img, patches

    except Exception as e:
        import traceback
        msg = f"Error: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        return msg, None, []


# -------------------------
# 构建 Markdown（包含整图与碎图，碎图以 base64 嵌入）
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


def build_markdown_with_patches(readable_text: str, result_img, patches_images=None, patches_base64=None,
                                embed_base64=True, max_width=1200, patch_max_width=600, quality=85):
    if patches_base64 is None and patches_images:
        patches_base64 = images_to_base64_list(patches_images, max_width=patch_max_width, quality=quality)
    md = []
    md.append("# OCR 结果\n")
    md.append("## 文本\n")
    md.append(readable_text or "*未识别到文本*")
    md.append("\n---\n")
    md.append("## OCR 可视化\n")
    if result_img is not None:
        if embed_base64:
            data_uri = pil_image_to_base64_datauri(result_img, max_width=max_width, quality=quality)
            if data_uri:
                md.append(f"![OCR 全图]({data_uri})\n")
            else:
                md.append("![OCR 全图](ocr_result.jpg)\n")
        else:
            md.append("![OCR 全图](ocr_result.jpg)\n")
    else:
        md.append("_无 OCR 可视化图_\n")
    md.append("\n---\n")
    md.append(f"## 碎图（共 {len(patches_base64) if patches_base64 else (len(patches_images) if patches_images else 0)} 张）\n")
    if not patches_base64:
        md.append("_无碎图_\n")
    else:
        for idx, uri in enumerate(patches_base64, start=1):
            md.append(f"### 碎图 {idx}\n")
            md.append(f"![patch_{idx}]({uri})\n\n")
    return "\n".join(md)


# -------------------------
# 导出 Markdown（支持嵌入或不嵌入）
# -------------------------
def export_markdown_with_patches(markdown_text: str, result_img, patches, embed_base64=True, quality=85):
    try:
        temp_dir = tempfile.mkdtemp()
        md_path = os.path.join(temp_dir, "result.md")
        if not embed_base64:
            if result_img is not None:
                try:
                    result_img.save(os.path.join(temp_dir, "ocr_result.jpg"), quality=quality)
                except Exception:
                    pass
            for idx, p in enumerate(patches, start=1):
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
# Gradio 界面（简洁布局）
# -------------------------
def create_demo():
    with gr.Blocks(title="DeepSeek-OCR", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🔍 DeepSeek-OCR\n上传图片并生成可读文本、结果图与碎图，支持导出 Markdown（碎图以 base64 嵌入）")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📤 Input & Settings")
                image_input = gr.Image(label="Upload Image", type="pil", sources=["upload", "clipboard"])
                prompt_type = gr.Radio(choices=["Free OCR", "Markdown Conversion", "Custom"],
                                       value="Markdown Conversion", label="Prompt Type")
                custom_prompt = gr.Textbox(label="Custom Prompt (if selected)", placeholder="Enter custom prompt...", lines=2, visible=False)
                model_size = gr.Radio(choices=["Tiny", "Small", "Base", "Large", "Gundam (Recommended)"],
                                      value="Gundam (Recommended)", label="Model Size")
                process_btn = gr.Button("🚀 Process Image", variant="primary")

                # 控制 custom_prompt 可见性
                def update_prompt_visibility(choice):
                    return gr.update(visible=(choice == "Custom"))
                prompt_type.change(fn=update_prompt_visibility, inputs=[prompt_type], outputs=[custom_prompt])

            with gr.Column(scale=1):
                gr.Markdown("### 📄 Results")
                output_text = gr.Textbox(label="Extracted Text (readable)", lines=20, max_lines=60, show_copy_button=True)
                result_image = gr.Image(label="Result Image (OCR visualization)", type="pil")
                patches_gallery = gr.Gallery(label="碎图 (patches / crops)", columns=6, type="pil")

                gr.Markdown("### 📝 Markdown & Export")
                readable_toggle = gr.Checkbox(label="将 LaTeX 转换为可读文本（pylatexenc）", value=True)
                embed_toggle = gr.Checkbox(label="在 Markdown 中嵌入图片（Base64）", value=True)
                generate_md_btn = gr.Button("📝 生成 Markdown")
                md_preview = gr.Markdown(label="Markdown 预览", value="")
                export_md_btn = gr.Button("💾 导出 Markdown (.md)")
                md_file = gr.File(label="下载生成的 Markdown 文件", interactive=False)

        # Process -> 返回文本, 整图, patches
        process_btn.click(
            fn=process_image_full,
            inputs=[image_input, prompt_type, custom_prompt, model_size],
            outputs=[output_text, result_image, patches_gallery]
        )

        # 生成 Markdown（将 patches 以 base64 嵌入）
        def to_md(text_result, result_img, patches_list, use_readable, embed_base64):
            # process_image_full 已返回可读文本（latex 已转换）；这里直接使用 text_result
            patches_images = patches_list or []
            md = build_markdown_with_patches(text_result, result_img, patches_images=patches_images,
                                             patches_base64=None, embed_base64=embed_base64)
            return md

        generate_md_btn.click(
            fn=to_md,
            inputs=[output_text, result_image, patches_gallery, readable_toggle, embed_toggle],
            outputs=[md_preview]
        )

        # 导出 Markdown（返回 md 文件路径以供下载）
        def export_md(md_str, result_img, patches_list, embed_base64):
            patches_images = patches_list or []
            md_path = export_markdown_with_patches(md_str, result_img, patches_images, embed_base64=embed_base64)
            return md_path

        export_md_btn.click(
            fn=export_md,
            inputs=[md_preview, result_image, patches_gallery, embed_toggle],
            outputs=[md_file]
        )

    return demo


if __name__ == "__main__":
    demo = create_demo()
    demo.launch(server_name="0.0.0.0", server_port=2714, share=False)
