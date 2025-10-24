# 确保安装了gradio，用pip install gradio

import gradio as gr
from transformers import AutoModel, AutoTokenizer
import torch
import os
from PIL import Image
import tempfile
import shutil

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
# 新增：LaTeX 转文本与 Markdown 构建/导出工具函数
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


def build_markdown_with_image(readable_text: str, image_obj) -> str:
    """
    生成包含文本与图片的 Markdown。
    - readable_text: 已转换为可读文本的结果
    - image_obj: 来自 gr.Image 的 PIL Image 或者路径
    策略：
      1) 至少嵌入用户上传的图片，满足“如果处理的图片中包含图片，返回markdown中需要包含图片”。
      2) 若后续需要插入文档内部图片，请在 infer 阶段保留图片列表与位置信息再扩展。
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
    # 由导出函数写入 input_image.jpg，在此使用固定相对路径
    md_parts.append("![输入图片](input_image.jpg)")
    md_parts.append("")
    return "\n".join(md_parts)


def export_markdown(markdown_text: str, image_obj):
    """
    将 Markdown 与图片导出到本地临时目录，并返回 .md 文件路径用于下载。
    - 在临时目录写入 input_image.jpg 和 result.md
    - Gradio 的 File 组件接收 .md 文件路径以供下载
    """
    try:
        temp_dir = tempfile.mkdtemp()
        md_path = os.path.join(temp_dir, "result.md")
        img_path = os.path.join(temp_dir, "input_image.jpg")

        # 保存图片
        if image_obj is not None:
            if isinstance(image_obj, str) and os.path.exists(image_obj):
                shutil.copy(image_obj, img_path)
            else:
                # 尝试作为 PIL.Image 保存
                try:
                    image_obj.save(img_path)
                except Exception:
                    # 非 PIL.Image 类型，则不保存图片
                    pass

        # 写入 Markdown
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(markdown_text or "")

        return md_path
    except Exception as e:
        # 返回 None 让 File 不显示，同时可在 UI 上提示
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
                generate_md_btn = gr.Button("📝 生成 Markdown", variant="secondary")
                md_preview = gr.Markdown(label="Markdown 预览", value="")
                export_md_btn = gr.Button("💾 导出 Markdown", variant="secondary")
                md_file = gr.File(label="下载生成的 Markdown 文件", interactive=False)

                gr.Markdown(
                    """
                    ### 📥 Export
                    1) 点击“生成 Markdown”预览文本与图片
                    2) 点击“导出 Markdown”保存至本地临时目录并下载 .md 文件（包含图片）
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
        def to_readable_and_md(text_result, image_obj, use_readable):
            """
            将文本（可能为 LaTeX）转换为可读文本，并生成 Markdown 预览。
            """
            try:
                readable = latex_to_readable_text(text_result) if use_readable else text_result
                md_str = build_markdown_with_image(readable, image_obj)
                return md_str
            except Exception as e:
                return f"生成 Markdown 失败：{e}"

        generate_md_btn.click(
            fn=to_readable_and_md,
            inputs=[output_text, image_input, readable_toggle],
            outputs=[md_preview]
        )

        # 导出 Markdown 逻辑
        def on_export_md(md_str, image_obj):
            """
            导出 Markdown 到临时目录并返回下载文件。
            """
            file_path = export_markdown(md_str, image_obj)
            return file_path

        export_md_btn.click(
            fn=on_export_md,
            inputs=[md_preview, image_input],
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
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
