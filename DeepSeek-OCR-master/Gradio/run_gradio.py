# 确保安装了gradio，用pip install gradio

import gradio as gr
from transformers import AutoModel, AutoTokenizer
import torch
import os
from PIL import Image
import tempfile
import shutil

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


def create_demo():
    """Create Gradio interface"""

    with gr.Blocks(title="DeepSeek-OCR Demo", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 🔍 DeepSeek-OCR Demo

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

                gr.Markdown(
                    """
                    ### 📥 Export
                    You can copy the results using the copy button above.
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
            - **Gundam**: Best balance with crop mode (1024x640 with cropping)

            **Note:** First run will download the model (~several GB). Requires CUDA-enabled GPU.
            """
        )

    return demo


if __name__ == "__main__":
    # Set CUDA device
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    # Create and launch demo
    demo = create_demo()
    demo.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,
        share=False,  # Set to True to create a public link
        debug=True
    )
