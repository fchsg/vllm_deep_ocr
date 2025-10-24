# run_gradio.py
# 注意：替换本文件会启动 Gradio 服务，确保依赖已安装
# pip install gradio transformers torch pillow pylatexenc

import gradio as gr
from transformers import AutoModel, AutoTokenizer
import torch
import os
from PIL import Image
import tempfile
import shutil
import io
import base64

# pylatexenc 用于 LaTeX 转文本
try:
    from pylatexenc.latex2text import LatexNodes2Text
except Exception as _e:
    print("pylatexenc 未安装，请先执行: pip install pylatexenc")

# Global variables for model and tokenizer
model = None
tokenizer = None


def load_model():
    """Load the DeepSeek-OCR model and tokenizer"""
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
        model = model.eval().cuda().to(torch.bfloat16)
        print("Model loaded successfully!")

    return model, tokenizer


def latex_to_readable_text(latex_str: str) -> str:
    """
    使用 pylatexenc 将 LaTeX 转换为可读纯文本。
    如果 pylatexenc 不可用或转换失败，返回原文。
    """
    if not latex_str or not latex_str.strip():
        return latex_str
    try:
        return LatexNodes2Text().latex_to_text(latex_str)
    except Exception:
        return latex_str


def pil_image_to_base64_datauri(img: Image.Image, max_width=1200, quality=85, fmt="JPEG"):
    """
    将 PIL Image 转为 base64 data URI（默认 JPEG）。
    - max_width: 若图片宽度大于该值，将按比例缩放
    - quality: JPEG 压缩质量（0-100）
    返回 data uri 字符串，例如 "data:image/jpeg;base64,...."
    """
    if img is None:
        return None
    try:
        w, h = img.size
    except Exception:
        return None

    # 缩放（仅宽度）
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
            # 使用 alpha 通道作为 mask
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


def build_markdown_with_image(readable_text: str, image_obj, embed_base64=True, max_width=1200, quality=85):
    """
    生成包含 OCR 文本与 OCR 结果图片的 Markdown。
    - readable_text: OCR 可读文本
    - image_obj: PIL.Image 或 路径
    - embed_base64: 是否将图片以 data URI 形式嵌入 Markdown（默认 True）
    """
    md_parts = []
    md_parts.append("# OCR 结果")
    md_parts.append("")
    md_parts.append("## 文本")
    md_parts.append("")
    md_parts.append(readable_text if readable_text else "*未识别到文本*")
    md_parts.append("")
    md_parts.append("## OCR 结果图片")
    md_parts.append("")

    if image_obj is None:
        md_parts.append("_无 OCR 输出图片_")
    else:
        if embed_base64:
            try:
                if isinstance(image_obj, str) and os.path.exists(image_obj):
                    pil_img = Image.open(image_obj)
                else:
                    pil_img = image_obj  # 期望是 PIL.Image
                data_uri = pil_image_to_base64_datauri(pil_img, max_width=max_width, quality=quality)
                if data_uri:
                    md_parts.append(f"![OCR 结果]({data_uri})")
                else:
                    md_parts.append("![OCR 结果](input_image.jpg)")
            except Exception:
                md_parts.append("![OCR 结果](input_image.jpg)")
        else:
            md_parts.append("![OCR 结果](input_image.jpg)")

    md_parts.append("")
    return "\n".join(md_parts)


def export_markdown(markdown_text: str, image_obj, embed_base64=True, max_width=1200, quality=85):
    """
    导出 Markdown，并返回可供 gr.File 下载的路径。
    - embed_base64 True: .md 中已包含图片 (data URI)
    - embed_base64 False: 会将图片保存为 input_image.jpg 与 result.md 同目录
    返回 md 文件的绝对路径或 None
    """
    try:
        temp_dir = tempfile.mkdtemp()
        md_path = os.path.join(temp_dir, "result.md")

        if not embed_base64 and image_obj is not None:
            img_path = os.path.join(temp_dir, "input_image.jpg")
            try:
                if isinstance(image_obj, str) and os.path.exists(image_obj):
                    shutil.copy(image_obj, img_path)
                else:
                    image_obj.save(img_path)
            except Exception as e:
                print(f"保存图片失败: {e}")

        with open(md_path, "w", encoding="utf-8") as f:
            f.write(markdown_text or "")

        return md_path
    except Exception as e:
        print(f"导出失败: {e}")
        return None


def find_result_image_in_dir(dir_path):
    """
    在模型输出目录中查找代表 OCR 结果的图片文件。
    策略：
      - 搜索常见图像扩展 (.png, .jpg, .jpeg)
      - 优先选择文件名包含 'result' 或 'vis' 或 'output' 的文件
      - 否则返回第一个找到的图像
    返回：PIL.Image 对象或 None
    """
    exts = ('.png', '.jpg', '.jpeg', '.webp')
    candidates = []
    for fname in os.listdir(dir_path):
        if fname.lower().endswith(exts):
            candidates.append(fname)

    if not candidates:
        return None

    # 优先级排序
    preferred_keywords = ['result', 'vis', 'output', 'pred', 'ocr']
    def score(name):
        n = name.lower()
        s = 0
        for i, kw in enumerate(preferred_keywords):
            if kw in n:
                s += (len(preferred_keywords) - i) * 10
        # 略微优先较大的文件（可能包含可视化）
        try:
            p = os.path.getsize(os.path.join(dir_path, name))
            s += int(p / 1024)  # 大文件得分更高
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


def process_image(image, prompt_type, custom_prompt, model_size):
    """
    Process image with OCR and return:
      - readable_text: 使用 pylatexenc 转换后的可读文本（直接显示在 Results）
      - result_img: PIL.Image（OCR 结果可视化图片或原始上传图片）
    """
    try:
        model, tokenizer = load_model()

        temp_dir = tempfile.mkdtemp()

        # 保存上传的输入图像
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

        # 捕获 stdout
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

        # 尝试读取输出目录中的文本文件（优先）
        ocr_text = ""
        for filename in os.listdir(temp_dir):
            if filename.endswith('.txt'):
                try:
                    with open(os.path.join(temp_dir, filename), 'r', encoding='utf-8') as f:
                        ocr_text += f.read() + "\n"
                except Exception:
                    pass

        # 如果没有文本文件，解析 captured_text
        if not ocr_text.strip() and captured_text.strip():
            lines = captured_text.split('\n')
            clean_lines = []
            for line in lines:
                if '<|ref|>' in line or '<|det|>' in line or '<|/ref|>' in line or '<|/det|>' in line:
                    import re
                    text_match = re.search(r'<\|/ref\|>(.*?)<\|det\|>', line)
                    if text_match:
                        clean_lines.append(text_match.group(1).strip())
                elif line.startswith('=====') or 'BASE:' in line or 'PATCHES:' in line or line.startswith('image:') or line.startswith('other:'):
                    continue
                elif line.strip():
                    clean_lines.append(line.strip())
            ocr_text = "\n".join(clean_lines)

        if not ocr_text.strip():
            # 如果 result 是字符串，可能直接返回文本
            if isinstance(result, str):
                ocr_text = result
            else:
                ocr_text = ""

        # 将 OCR 文本（可能为 LaTeX）转换为可读文本
        readable = latex_to_readable_text(ocr_text) if ocr_text else ""

        # 查找模型生成的结果图片（优先），若无则使用输入图片
        result_img = find_result_image_in_dir(temp_dir)
        if result_img is None:
            result_img = input_pil

        # 将图片加载到内存后可以删除临时目录
        # result_img 已经是 PIL.Image 对象
        try:
            shutil.rmtree(temp_dir)
        except Exception:
            pass

        # 返回 readable text 与 PIL image
        return readable if readable.strip() else "No text detected in image.", result_img

    except Exception as e:
        import traceback
        msg = f"Error: {str(e)}\n\nTraceback:\n{traceback.format_exc()}\n\nPlease make sure you have a CUDA-enabled GPU and all dependencies installed."
        # 返回错误信息与空图片
        return msg, None


def create_demo():
    """Create Gradio interface"""

    with gr.Blocks(title="DeepSeek-OCR Demo", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 🔍 DeepSeek-OCR

            Upload an image containing text, documents, charts, or tables to extract text using DeepSeek-OCR.

            **Features:**
            - Free OCR for general text extraction
            - Markdown conversion for document structure
            - Multiple model sizes for different accuracy/speed tradeoffs
            - Support for various document types
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                # Input section
                gr.Markdown("### 📤 Input")
                image_input = gr.Image(
                    label="Upload Image",
                    type="pil",
                    sources=["upload", "clipboard"]
                )

                gr.Markdown("### ⚙️ Settings")

                prompt_type = gr.Radio(
                    choices=["Free OCR", "Markdown Conversion", "Custom"],
                    value="Markdown Conversion",
                    label="Prompt Type",
                    info="Choose the type of OCR processing"
                )

                custom_prompt = gr.Textbox(
                    label="Custom Prompt (if selected)",
                    placeholder="Enter your custom prompt here...",
                    lines=2,
                    visible=False
                )

                model_size = gr.Radio(
                    choices=[
                        "Tiny",
                        "Small",
                        "Base",
                        "Large",
                        "Gundam (Recommended)"
                    ],
                    value="Gundam (Recommended)",
                    label="Model Size",
                    info="Larger models are more accurate but slower"
                )

                process_btn = gr.Button("🚀 Process Image", variant="primary", size="lg")

                gr.Markdown(
                    """
                    ### 💡 Tips
                    - **Gundam** mode works best for most documents
                    - Use **Markdown Conversion** for structured documents
                    - **Free OCR** for simple text extraction
                    - Higher resolution images give better results
                    """
                )

            with gr.Column(scale=1):
                # Output section
                gr.Markdown("### 📄 Results")
                output_text = gr.Textbox(
                    label="Extracted Text (readable)",
                    lines=20,
                    max_lines=30,
                    show_copy_button=True
                )
                result_image = gr.Image(
                    label="Result Image (OCR output)",
                    type="pil"
                )

                # 新增：LaTeX 转文本 + Markdown 生成/导出
                gr.Markdown("### 📝 Markdown")
                readable_toggle = gr.Checkbox(
                    label="将 LaTeX 转换为可读文本（pylatexenc）",
                    value=True,
                    info="开启后将使用 pylatexenc 把 LaTeX 转为普通文本（Results 已经返回可读文本）"
                )
                embed_toggle = gr.Checkbox(
                    label="在 Markdown 中嵌入图片（Base64）",
                    value=True,
                    info="开启后图片会以 base64 data URI 嵌入 Markdown，前端可以直接预览"
                )
                generate_md_btn = gr.Button("📝 生成 Markdown", variant="secondary")
                md_preview = gr.Markdown(label="Markdown 预览", value="")
                export_md_btn = gr.Button("💾 导出 Markdown", variant="secondary")
                md_file = gr.File(label="下载生成的 Markdown 文件", interactive=False)

                gr.Markdown(
                    """
                    ### 📥 Export
                    1) 点击“Process Image”得到 Results（可读文本 + OCR 图像）
                    2) 点击“生成 Markdown”预览文本与图片
                    3) 点击“导出 Markdown”保存并下载 .md 文件（包含图片或与图片配套）
                    """
                )

        # Show/hide custom prompt based on selection
        def update_prompt_visibility(choice):
            return gr.update(visible=(choice == "Custom"))

        prompt_type.change(
            fn=update_prompt_visibility,
            inputs=[prompt_type],
            outputs=[custom_prompt]
        )

        # Process button click -> return readable text and result image
        process_btn.click(
            fn=process_image,
            inputs=[image_input, prompt_type, custom_prompt, model_size],
            outputs=[output_text, result_image]
        )

        # 生成 Markdown 逻辑 (使用 process 返回的文本与图片)
        def to_readable_and_md(text_result, image_obj, use_readable, embed_base64):
            """
            将文本（Results 中的文本）和 OCR 结果图片生成 Markdown 预览。
            Note: text_result 在 process_image 中已为可读文本。
            """
            try:
                # 如果用户关闭了 pylatexenc，但 process_image 已经转了，直接使用 text_result
                readable = text_result
                md_str = build_markdown_with_image(readable, image_obj, embed_base64=embed_base64)
                return md_str
            except Exception as e:
                return f"生成 Markdown 失败：{e}"

        generate_md_btn.click(
            fn=to_readable_and_md,
            inputs=[output_text, result_image, readable_toggle, embed_toggle],
            outputs=[md_preview]
        )

        # 导出 Markdown 逻辑
        def on_export_md(md_str, image_obj, embed_base64):
            file_path = export_markdown(md_str, image_obj, embed_base64=embed_base64)
            return file_path

        export_md_btn.click(
            fn=on_export_md,
            inputs=[md_preview, result_image, embed_toggle],
            outputs=[md_file]
        )

        # Add examples
        gr.Markdown("### 📚 Example Images")
        gr.Examples(
            examples=[
                ["example_document.jpg", "Markdown Conversion", "", "Gundam (Recommended)"],
                ["example_receipt.jpg", "Free OCR", "", "Small"],
            ],
            inputs=[image_input, prompt_type, custom_prompt, model_size],
            outputs=[output_text, result_image],
            fn=process_image,
            cache_examples=False,
        )

        gr.Markdown(
            """
            ---
            ### ℹ️ About

            This demo uses [DeepSeek-OCR](https://huggingface.co/deepseek-ai/DeepSeek-OCR) for optical character recognition.

            **Model Sizes Explained:**
            - **Tiny**: Fastest, lowest accuracy (512x512)
            - **Small**: Fast, good for simple documents (640x640)
            - **Base**: Balanced performance (1024x1024)
            - **Large**: High accuracy, slower (1280x1280)
            - **Gundam (Recommended)**: Balanced config with crop mode for complex docs
            """
        )

    return demo


if __name__ == "__main__":
    demo = create_demo()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
