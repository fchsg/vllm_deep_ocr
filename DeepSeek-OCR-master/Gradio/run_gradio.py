# run_gradio.py
# 确保安装了 gradio，用 pip install gradio
# 确保安装 pylatexenc 用于 LaTeX 转文本：pip install pylatexenc

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


def process_image(image, prompt_type, custom_prompt, model_size):
    """Process image with OCR"""
    try:
        # Load model if not already loaded
        model, tokenizer = load_model()

        # Create temporary directory for output
        temp_dir = tempfile.mkdtemp()

        # Save uploaded image temporarily
        temp_image_path = os.path.join(temp_dir, "input_image.jpg")
        if isinstance(image, str):
            shutil.copy(image, temp_image_path)
        else:
            image.save(temp_image_path)

        # Set prompt based on selection
        if prompt_type == "Free OCR":
            prompt = "<image>\nFree OCR. "
        elif prompt_type == "Markdown Conversion":
            prompt = "<image>\n<|grounding|>Convert the document to markdown. "
        elif prompt_type == "Custom":
            prompt = f"<image>\n{custom_prompt}"
        else:
            prompt = "<image>\nFree OCR. "

        # Set model size parameters
        size_configs = {
            "Tiny": {"base_size": 512, "image_size": 512, "crop_mode": False},
            "Small": {"base_size": 640, "image_size": 640, "crop_mode": False},
            "Base": {"base_size": 1024, "image_size": 1024, "crop_mode": False},
            "Large": {"base_size": 1280, "image_size": 1280, "crop_mode": False},
            "Gundam (Recommended)": {"base_size": 1024, "image_size": 640, "crop_mode": True}
        }

        config = size_configs[model_size]

        # Capture stdout to get the OCR results
        import sys
        from io import StringIO

        # Redirect stdout to capture print statements
        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()

        try:
            # Run inference
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
            # Restore stdout
            sys.stdout = old_stdout

        # Get captured output
        captured_text = captured_output.getvalue()

        # Try to read from saved text file if it exists
        ocr_text = ""
        for filename in os.listdir(temp_dir):
            if filename.endswith('.txt'):
                with open(os.path.join(temp_dir, filename), 'r', encoding='utf-8') as f:
                    ocr_text += f.read() + "\n"

        # If we found text in files, use that; otherwise use captured output
        if ocr_text.strip():
            final_result = ocr_text.strip()
        elif captured_text.strip():
            # Parse the captured output to extract actual OCR text
            # Remove detection boxes and reference tags
            lines = captured_text.split('\n')
            clean_lines = []
            for line in lines:
                # Skip lines with detection boxes and reference tags
                if '<|ref|>' in line or '<|det|>' in line or '<|/ref|>' in line or '<|/det|>' in line:
                    # Extract text between tags
                    import re
                    # Pattern to match text between </ref|> and <|det|>
                    text_match = re.search(r'<\|/ref\|>(.*?)<\|det\|>', line)
                    if text_match:
                        clean_lines.append(text_match.group(1).strip())
                elif line.startswith('=====') or 'BASE:' in line or 'PATCHES:' in line or line.startswith(
                        'image:') or line.startswith('other:'):
                    continue
                elif line.strip():
                    clean_lines.append(line.strip())

            final_result = '\n'.join(clean_lines)
        elif isinstance(result, str):
            final_result = result
        else:
            final_result = str(result) if result else "No text detected in image."

        # Clean up temporary directory
        shutil.rmtree(temp_dir)

        return final_result if final_result.strip() else "No text detected in image."

    except Exception as e:
        import traceback
        return f"Error: {str(e)}\n\nTraceback:\n{traceback.format_exc()}\n\nPlease make sure you have a CUDA-enabled GPU and all dependencies installed."


# =========================
# 新增：LaTeX 转文本与 Markdown 构建/导出工具函数（含 base64 嵌入）
# =========================

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
    # 确保为 PIL.Image
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
    # 对 PNG 或透明图片，若需要保留透明度可使用 PNG
    save_kwargs = {}
    if img_format == "JPEG":
        # convert to RGB to avoid 保存 RGBA 导致错误
        if img.mode in ("RGBA", "LA"):
            background = Image.new("RGB", img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[3])  # 3 is alpha
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
    生成包含文本与图片的 Markdown。
    - 如果 embed_base64=True，会将输入图片编码为 base64 data URI 并直接嵌入 Markdown。
    - 否则使用相对路径 'input_image.jpg'（需配合导出时打包图片）。
    """
    md_parts = []
    md_parts.append("# OCR 结果")
    md_parts.append("")
    md_parts.append("## 文本")
    md_parts.append("")
    md_parts.append(readable_text if readable_text else "*未识别到文本*")
    md_parts.append("")
    md_parts.append("## 图片")
    md_parts.append("")

    if image_obj is None:
        md_parts.append("_无上传图片_")
    else:
        if embed_base64:
            # 如果 image_obj 是路径则先打开
            try:
                if isinstance(image_obj, str) and os.path.exists(image_obj):
                    pil_img = Image.open(image_obj)
                else:
                    pil_img = image_obj  # 期望是 PIL.Image
                data_uri = pil_image_to_base64_datauri(pil_img, max_width=max_width, quality=quality)
                if data_uri:
                    md_parts.append(f"![输入图片]({data_uri})")
                else:
                    md_parts.append("![输入图片](input_image.jpg)")
            except Exception:
                md_parts.append("![输入图片](input_image.jpg)")
        else:
            md_parts.append("![输入图片](input_image.jpg)")

    md_parts.append("")
    return "\n".join(md_parts)


def export_markdown(markdown_text: str, image_obj, embed_base64=True, max_width=1200, quality=85):
    """
    将 Markdown 与图片导出到本地临时目录，并返回 .md 文件路径用于下载。
    - 当 embed_base64=True 时，Markdown 中已嵌入图片，无需单独保存图片（.md 即包含图像）
    - 当 embed_base64=False 时，会把图片保存为 input_image.jpg 与 result.md 同目录
    注意：将 base64 嵌入 .md 会使文件更大，但能保证 Gradio 前端预览正常。
    """
    try:
        temp_dir = tempfile.mkdtemp()
        md_path = os.path.join(temp_dir, "result.md")

        # 如果不嵌入，需要把图片存为 input_image.jpg
        if not embed_base64 and image_obj is not None:
            img_path = os.path.join(temp_dir, "input_image.jpg")
            try:
                if isinstance(image_obj, str) and os.path.exists(image_obj):
                    shutil.copy(image_obj, img_path)
                else:
                    image_obj.save(img_path)
            except Exception as e:
                print(f"保存图片失败: {e}")

        # 写入 Markdown
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(markdown_text or "")

        return md_path
    except Exception as e:
        print(f"导出失败: {e}")
        return None


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
                    label="Extracted Text",
                    lines=20,
                    max_lines=30,
                    show_copy_button=True
                )

                # 新增：LaTeX 转文本 + Markdown 生成/导出
                gr.Markdown("### 📝 Markdown")
                readable_toggle = gr.Checkbox(
                    label="将 LaTeX 转换为可读文本（pylatexenc）",
                    value=True,
                    info="开启后将使用 pylatexenc 把 LaTeX 转为普通文本"
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
                    1) 点击“生成 Markdown”预览文本与图片
                    2) 点击“导出 Markdown”保存至本地临时目录并下载 .md 文件（包含图片或与图片配套）
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

        # Process button click
        process_btn.click(
            fn=process_image,
            inputs=[image_input, prompt_type, custom_prompt, model_size],
            outputs=[output_text]
        )

        # 生成 Markdown 逻辑
        def to_readable_and_md(text_result, image_obj, use_readable, embed_base64):
            """
            将文本（可能为 LaTeX）转换为可读文本，并生成 Markdown 预览（默认嵌入 base64）。
            """
            try:
                readable = latex_to_readable_text(text_result) if use_readable else text_result
                md_str = build_markdown_with_image(readable, image_obj, embed_base64=embed_base64)
                return md_str
            except Exception as e:
                return f"生成 Markdown 失败：{e}"

        generate_md_btn.click(
            fn=to_readable_and_md,
            inputs=[output_text, image_input, readable_toggle, embed_toggle],
            outputs=[md_preview]
        )

        # 导出 Markdown 逻辑
        def on_export_md(md_str, image_obj, embed_base64):
            """
            导出 Markdown 到临时目录并返回下载文件。
            """
            file_path = export_markdown(md_str, image_obj, embed_base64=embed_base64)
            return file_path

        export_md_btn.click(
            fn=on_export_md,
            inputs=[md_preview, image_input, embed_toggle],
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
            outputs=[output_text],
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
    # 允许本地访问，可按需修改 server_name/port
    demo.launch(server_name="0.0.0.0", server_port=2714, share=False)
