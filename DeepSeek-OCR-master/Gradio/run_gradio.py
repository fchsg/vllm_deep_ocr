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
        # 尝试 GPU+bfloat16，若失败回退 CPU+float32
        try:
            model = model.eval().cuda().to(torch.bfloat16)
        except Exception:
            model = model.eval().to(torch.float32)
        print("Model loaded successfully!")
    return model, tokenizer


# -------------------------
# PIL -> base64 data URI 辅助
# -------------------------
def pil_image_to_base64_datauri(img: Image.Image, max_width=1200, quality=85, fmt="JPEG"):
    """将 PIL.Image 转为 base64 data URI（缩放 + 压缩以控制大小）"""
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
# 找到 temp_dir 中的图片（整图与碎图）
# -------------------------
def collect_images_from_output_dir(output_dir, max_patches=32):
    """
    在模型输出目录中查找图片文件，分为：
      - overall_images: 可能的可视化整图（选优先级高的）
      - patch_images: 可能的碎图 / crops / patches
    返回： (overall_images_list[PIL.Image], patch_images_list[PIL.Image])
    """
    overall_images = []
    patch_images = []

    if not output_dir or not os.path.exists(output_dir):
        return overall_images, patch_images

    exts = ('.png', '.jpg', '.jpeg', '.webp')
    keywords_patch = ['patch', 'patches', 'crop', 'crops', 'fragment', 'frag', 'cropbox', 'box', 'crop_img', 'patch_img']
    keywords_overall = ['result', 'vis', 'visual', 'output', 'pred', 'ocr', 'final']

    # 收集所有图片路径
    image_paths = []
    for root, dirs, files in os.walk(output_dir):
        for f in files:
            if f.lower().endswith(exts):
                image_paths.append(os.path.join(root, f))

    # 优先挑选带关键词的文件到对应列表
    for p in image_paths:
        name = os.path.basename(p).lower()
        if any(k in name for k in keywords_patch):
            patch_images.append(p)
        elif any(k in name for k in keywords_overall):
            overall_images.append(p)
        else:
            # 无明显关键词，暂先作为 patch 候选
            patch_images.append(p)

    # 去重并按优先级排序（整体图按 size 或关键字得分）
    def score_overall(path):
        n = os.path.basename(path).lower()
        s = 0
        for i, kw in enumerate(keywords_overall):
            if kw in n:
                s += (len(keywords_overall) - i) * 10
        try:
            s += int(os.path.getsize(path) / 1024)
        except Exception:
            pass
        return s

    overall_images = sorted(set(overall_images), key=lambda x: score_overall(x), reverse=True)
    # patch_images 保持找到顺序，但去重
    patch_images = list(dict.fromkeys(patch_images))

    # 读取为 PIL.Image 对象（整体图只取第一张作为“整体可视化图”）
    overall_pils = []
    patch_pils = []

    for p in overall_images:
        try:
            overall_pils.append(Image.open(p).convert("RGB"))
        except Exception:
            continue

    for p in patch_images:
        try:
            patch_pils.append(Image.open(p).convert("RGB"))
        except Exception:
            continue
        if len(patch_pils) >= max_patches:
            break

    return overall_pils, patch_pils


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
# 主流程：运行模型推理并在 temp_dir 中收集模型生成的图片
# -------------------------
def process_image_and_collect_output_images(image, prompt_type, custom_prompt, model_size):
    """
    运行模型推理并返回：
      - readable_text: OCR 文本（LaTeX 已转换为可读文本）
      - patch_images: list[PIL.Image]（模型输出目录中的碎图 / crops）
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

        # 捕获 stdout 并运行 model.infer（许多实现会把结果写入 output_path）
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

        # 尝试读取 temp_dir 下的文本文件作为 OCR 文本
        ocr_text = ""
        for filename in os.listdir(temp_dir):
            if filename.endswith('.txt'):
                try:
                    with open(os.path.join(temp_dir, filename), 'r', encoding='utf-8') as f:
                        ocr_text += f.read() + "\n"
                except Exception:
                    pass

        # 若无 txt，则解析 captured_text（兼容老实现）
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

        # 在 temp_dir 中收集模型生成的图片（整图与碎图）
        overall_imgs, patch_imgs = collect_images_from_output_dir(temp_dir, max_patches=48)

        # If model outputs only overall visualizations, but you want patches, you could
        # implement bbox parsing and cropping here (requires model output format).
        # For now: treat patch_imgs as "识别为图片的内容" to be returned and embedded.

        # 清理临时目录（patch_imgs 已加载到内存）
        try:
            shutil.rmtree(temp_dir)
        except Exception:
            pass

        return readable if readable.strip() else "No text detected in image.", patch_imgs

    except Exception as e:
        import traceback
        msg = f"Error: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        return msg, []


# -------------------------
# 将 PIL 列表转换为 base64 列表
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
# 构建 Markdown：把所有 patch 图作为“识别为图片的内容”以 base64 嵌入
# -------------------------
def build_markdown_from_text_and_patches(readable_text: str, patches_images=None, patches_base64=None,
                                        embed_base64=True, patch_max_width=600, quality=85):
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
# 导出 Markdown（写入临时目录并返回 md 路径）
# -------------------------
def export_markdown_with_patches(markdown_text: str, patches_images, embed_base64=True, quality=85):
    try:
        temp_dir = tempfile.mkdtemp()
        md_path = os.path.join(temp_dir, "result.md")

        if not embed_base64:
            # 保存 patches 到目录，md 需引用相对路径 —— 此处通常我们默认 embed=True
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
# Gradio 界面（简洁布局）
# -------------------------
def create_demo():
    with gr.Blocks(title="DeepSeek-OCR (patches embedded)", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🔍 DeepSeek-OCR - 把模型生成的小碎图嵌入 Markdown\n上传图片 -> OCR -> 返回可读文本 + 识别为图片的碎图。生成 Markdown 时以 base64 嵌入这些碎图。")

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

        # 处理 -> 返回可读文本, patches 列表
        process_btn.click(
            fn=process_image_and_collect_output_images,
            inputs=[image_input, prompt_type, custom_prompt, model_size],
            outputs=[output_text, patches_gallery]
        )

        # 生成 Markdown（把 patches 转为 base64 嵌入）
        def to_md(text_result, patches_list, use_readable, embed_base64):
            readable = text_result or ""
            patches_images = patches_list or []
            md = build_markdown_from_text_and_patches(readable, patches_images=patches_images, patches_base64=None, embed_base64=embed_base64)
            return md

        generate_md_btn.click(
            fn=to_md,
            inputs=[output_text, patches_gallery, readable_toggle, embed_toggle],
            outputs=[md_preview]
        )

        # 导出 Markdown（返回 md 路径）
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
