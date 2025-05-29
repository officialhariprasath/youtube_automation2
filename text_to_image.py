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

# LoRA DLCs with images and info (add more as needed)
loras = [
    {"title": "Super Realism", "repo": "strangerzonehf/Flux-Super-Realism-LoRA", "weights": "super-realism.safetensors", "image": "https://huggingface.co/strangerzonehf/Flux-Super-Realism-LoRA/resolve/main/images/1.png"},
    {"title": "Dalle Mix", "repo": "prithivMLmods/Flux-Dalle-Mix-LoRA", "weights": "dalle-mix.safetensors", "image": "https://huggingface.co/prithivMLmods/Flux-Dalle-Mix-LoRA/resolve/main/images/D3.png"},
    {"title": "Ghibli Flux", "repo": "strangerzonehf/Ghibli-Flux-Cartoon-LoRA", "weights": "Ghibili-Cartoon-Art.safetensors", "image": "https://huggingface.co/strangerzonehf/Ghibli-Flux-Cartoon-LoRA/resolve/main/images/3333.png"},
    {"title": "Sketch_Smudge", "repo": "strangerzonehf/Flux-Sketch-Smudge-LoRA", "weights": "Sketch-Smudge.safetensors", "image": "https://huggingface.co/strangerzonehf/Flux-Sketch-Smudge-LoRA/resolve/main/images/5.png"},
    {"title": "Animeo Mix", "repo": "strangerzonehf/Flux-Animeo-v1-LoRA", "weights": "Animeo.safetensors", "image": "https://huggingface.co/strangerzonehf/Flux-Animeo-v1-LoRA/resolve/main/images/A4.png"},
    {"title": "Animex Mix", "repo": "strangerzonehf/Flux-Animex-v2-LoRA", "weights": "Animex.safetensors", "image": "https://huggingface.co/strangerzonehf/Flux-Animex-v2-LoRA/resolve/main/images/A33.png"},
]

pipe = None
selected_lora_idx = 0

def load_model():
    model_id = "stabilityai/stable-diffusion-2-1"
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
    return pipe

def add_custom_lora(custom_repo, custom_weights, custom_image_url, custom_title):
    loras.append({
        "title": custom_title or f"Custom LoRA {len(loras)+1}",
        "repo": custom_repo,
        "weights": custom_weights,
        "image": custom_image_url or ""
    })
    return get_lora_gallery(), gr.update(value="", visible=True), gr.update(value="", visible=True), gr.update(value="", visible=True), gr.update(value="", visible=True)

def get_lora_gallery():
    return [[lora["image"], lora["title"]] for lora in loras]

def select_lora(evt: gr.SelectData):
    global selected_lora_idx
    selected_lora_idx = evt.index
    return selected_lora_idx

def generate_image(prompt, negative_prompt, steps, guidance_scale, width, height, seed, lora_idx):
    global pipe
    if pipe is None:
        pipe = load_model()
    lora = loras[lora_idx]
    # (Optional) Download and apply LoRA weights here if needed
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
    gr.Markdown("# FLUX-LoRA-DLC Text-to-Image Generator (Hugging Face Style)")
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("**250+ LoRA DLC's**")
            lora_gallery = gr.Gallery(
                value=get_lora_gallery(),
                label="Select a LoRA DLC",
                elem_id="lora_gallery",
                show_label=False,
                columns=3,
                rows=2,
                height=320,
                allow_preview=False
            )
            custom_repo = gr.Textbox(label="Custom LoRA Repo (e.g. strangerzonehf/Flux-Super-Realism-LoRA)")
            custom_weights = gr.Textbox(label="Custom Weights Filename (e.g. super-realism.safetensors)")
            custom_image_url = gr.Textbox(label="Custom Image URL (optional)")
            custom_title = gr.Textbox(label="Custom LoRA Title (optional)")
            add_custom_btn = gr.Button("Add Custom LoRA")
        with gr.Column(scale=2):
            gr.Markdown("**Generated Image**")
            output = gr.Image(label="Generated Image", type="pil")
            prompt = gr.Textbox(label="Prompt", placeholder="Describe your image...")
            negative_prompt = gr.Textbox(label="Negative Prompt", placeholder="What do you NOT want?")
            steps = gr.Slider(1, 100, value=30, label="Steps")
            guidance_scale = gr.Slider(1, 20, value=7.5, label="Guidance Scale")
            width = gr.Slider(256, 1024, value=512, step=64, label="Width")
            height = gr.Slider(256, 1024, value=512, step=64, label="Height")
            seed = gr.Number(value=42, label="Seed")
            lora_idx = gr.State(0)
            generate_btn = gr.Button("Generate")
    lora_gallery.select(select_lora, None, lora_idx)
    generate_btn.click(
        generate_image,
        inputs=[prompt, negative_prompt, steps, guidance_scale, width, height, seed, lora_idx],
        outputs=output
    )
    add_custom_btn.click(
        add_custom_lora,
        inputs=[custom_repo, custom_weights, custom_image_url, custom_title],
        outputs=[lora_gallery, custom_repo, custom_weights, custom_image_url, custom_title]
    )

if __name__ == "__main__":
    demo.launch(share=True) 