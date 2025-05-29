import sys
import subprocess

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Install gradio and diffusers if running in Colab
try:
    import google.colab
    install('gradio')
    install('diffusers')
    install('torch')
    install('transformers')
    install('accelerate')
    install('safetensors')
    install('Pillow')
except ImportError:
    pass

import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import gradio as gr

def load_model():
    # Load the base model
    model_id = "stabilityai/stable-diffusion-2-1"
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16
    )
    
    # Use DPMSolverMultistepScheduler for better quality
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    
    # Move to GPU if available
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
    
    return pipe

pipe = None

def generate_image(prompt, negative_prompt, steps, guidance_scale, width, height, seed):
    global pipe
    if pipe is None:
        pipe = load_model()
    generator = torch.manual_seed(int(seed))
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=int(steps),
        guidance_scale=float(guidance_scale),
        width=int(width),
        height=int(height),
        generator=generator
    ).images[0]
    return image

with gr.Blocks() as demo:
    gr.Markdown("# Text-to-Image Generator (Stable Diffusion 2.1)")
    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(label="Prompt", placeholder="Describe your image...")
            negative_prompt = gr.Textbox(label="Negative Prompt", placeholder="What do you NOT want?")
            steps = gr.Slider(1, 100, value=30, label="Steps")
            guidance_scale = gr.Slider(1, 20, value=7.5, label="Guidance Scale")
            width = gr.Slider(256, 1024, value=512, step=64, label="Width")
            height = gr.Slider(256, 1024, value=512, step=64, label="Height")
            seed = gr.Number(value=42, label="Seed")
            generate_btn = gr.Button("Generate")
        with gr.Column():
            output = gr.Image(label="Generated Image", type="pil")
    generate_btn.click(
        generate_image,
        inputs=[prompt, negative_prompt, steps, guidance_scale, width, height, seed],
        outputs=output
    )

if __name__ == "__main__":
    demo.launch(share=True) 