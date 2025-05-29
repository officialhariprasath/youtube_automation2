# FLUX-LoRA-DLC Colab Script
# All features: LoRA DLC/style selection, text-to-image, img2img, advanced options, custom LoRA, Gradio UI

import os
import sys
import subprocess
import torch
import gradio as gr
from PIL import Image
import numpy as np
from typing import List, Dict, Any

# Install dependencies (Colab-friendly)
def install_deps():
    pkgs = [
        'git+https://github.com/huggingface/diffusers.git',
        'git+https://github.com/huggingface/transformers.git',
        'git+https://github.com/huggingface/accelerate.git',
        'git+https://github.com/huggingface/peft.git',
        'safetensors', 'sentencepiece', 'hf_xet', 'gradio', 'Pillow'
    ]
    for pkg in pkgs:
        subprocess.run([sys.executable, '-m', 'pip', 'install', pkg])

# Only install if running in Colab
def is_colab():
    try:
        import google.colab
        return True
    except ImportError:
        return False

if is_colab():
    install_deps()

import torch
from diffusers import DiffusionPipeline
from huggingface_hub import hf_hub_download

# LoRA DLCs (short list for demo, add more as needed)
loras = [
    {"title": "Super Realism", "repo": "strangerzonehf/Flux-Super-Realism-LoRA", "weights": "super-realism.safetensors", "trigger_word": "Super Realism"},
    {"title": "Dalle Mix", "repo": "prithivMLmods/Flux-Dalle-Mix-LoRA", "weights": "dalle-mix.safetensors", "trigger_word": "dalle-mix"},
    {"title": "Ghibli Flux", "repo": "strangerzonehf/Ghibli-Flux-Cartoon-LoRA", "weights": "Ghibili-Cartoon-Art.safetensors", "trigger_word": "Ghibli Art"},
    {"title": "Sketch_Smudge", "repo": "strangerzonehf/Flux-Sketch-Smudge-LoRA", "weights": "Sketch-Smudge.safetensors", "trigger_word": "Sketch Smudge"},
    {"title": "Animeo Mix", "repo": "strangerzonehf/Flux-Animeo-v1-LoRA", "weights": "Animeo.safetensors", "trigger_word": "Animeo"},
    {"title": "Animex Mix", "repo": "strangerzonehf/Flux-Animex-v2-LoRA", "weights": "Animex.safetensors", "trigger_word": "Animex"},
    # ... (add more from your app.py as needed)
]

# Allow adding custom LoRA DLCs
def add_custom_lora(title, repo, weights, trigger_word):
    loras.append({"title": title, "repo": repo, "weights": weights, "trigger_word": trigger_word})

def download_lora_weights(repo, weights):
    return hf_hub_download(repo_id=repo, filename=weights)

# Load base pipeline (Flux or SDXL, adjust as needed)
def load_pipeline():
    base_model = "stabilityai/stable-diffusion-2-1"  # or your preferred base
    pipe = DiffusionPipeline.from_pretrained(base_model, torch_dtype=torch.float16)
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
    return pipe

# Apply LoRA weights (using PEFT or diffusers)
def apply_lora(pipe, lora_path, lora_scale):
    # This is a placeholder; actual LoRA application may differ
    # For diffusers >=0.18.0, use pipe.load_lora_weights()
    pipe.load_lora_weights(lora_path, weight_name=None, adapter_name=None, scale=lora_scale)
    return pipe

# Text-to-Image
def generate_image(prompt, negative_prompt, steps, cfg_scale, width, height, seed, lora_idx, lora_scale):
    torch.manual_seed(seed)
    pipe = load_pipeline()
    lora = loras[lora_idx]
    lora_path = download_lora_weights(lora["repo"], lora["weights"])
    pipe = apply_lora(pipe, lora_path, lora_scale)
    image = pipe(prompt=prompt, negative_prompt=negative_prompt, num_inference_steps=steps, guidance_scale=cfg_scale, width=width, height=height).images[0]
    return image

# Img2Img
def generate_img2img(prompt, negative_prompt, init_image, denoise_strength, steps, cfg_scale, width, height, seed, lora_idx, lora_scale):
    torch.manual_seed(seed)
    pipe = load_pipeline()
    lora = loras[lora_idx]
    lora_path = download_lora_weights(lora["repo"], lora["weights"])
    pipe = apply_lora(pipe, lora_path, lora_scale)
    image = pipe(prompt=prompt, negative_prompt=negative_prompt, image=init_image, strength=denoise_strength, num_inference_steps=steps, guidance_scale=cfg_scale, width=width, height=height).images[0]
    return image

# Gradio UI
def gradio_ui():
    with gr.Blocks() as demo:
        gr.Markdown("# FLUX-LoRA-DLC (Colab Edition)")
        with gr.Tab("Text-to-Image"):
            prompt = gr.Textbox(label="Prompt")
            negative_prompt = gr.Textbox(label="Negative Prompt")
            steps = gr.Slider(1, 100, value=28, label="Steps")
            cfg_scale = gr.Slider(1, 20, value=3.5, label="CFG Scale")
            width = gr.Slider(256, 1024, value=512, step=64, label="Width")
            height = gr.Slider(256, 1024, value=512, step=64, label="Height")
            seed = gr.Number(value=42, label="Seed")
            lora_idx = gr.Dropdown(choices=[(l["title"], i) for i, l in enumerate(loras)], value=0, label="LoRA DLC/Style")
            lora_scale = gr.Slider(0, 3, value=1.0, label="LoRA Scale")
            btn = gr.Button("Generate")
            output = gr.Image(label="Generated Image")
            btn.click(generate_image, inputs=[prompt, negative_prompt, steps, cfg_scale, width, height, seed, lora_idx, lora_scale], outputs=output)
        with gr.Tab("Image-to-Image"):
            prompt2 = gr.Textbox(label="Prompt")
            negative_prompt2 = gr.Textbox(label="Negative Prompt")
            init_image = gr.Image(label="Input Image")
            denoise_strength = gr.Slider(0.1, 1.0, value=0.75, label="Denoise Strength")
            steps2 = gr.Slider(1, 100, value=28, label="Steps")
            cfg_scale2 = gr.Slider(1, 20, value=3.5, label="CFG Scale")
            width2 = gr.Slider(256, 1024, value=512, step=64, label="Width")
            height2 = gr.Slider(256, 1024, value=512, step=64, label="Height")
            seed2 = gr.Number(value=42, label="Seed")
            lora_idx2 = gr.Dropdown(choices=[(l["title"], i) for i, l in enumerate(loras)], value=0, label="LoRA DLC/Style")
            lora_scale2 = gr.Slider(0, 3, value=1.0, label="LoRA Scale")
            btn2 = gr.Button("Generate")
            output2 = gr.Image(label="Generated Image")
            btn2.click(generate_img2img, inputs=[prompt2, negative_prompt2, init_image, denoise_strength, steps2, cfg_scale2, width2, height2, seed2, lora_idx2, lora_scale2], outputs=output2)
        with gr.Tab("Add Custom LoRA"):
            title = gr.Textbox(label="Title")
            repo = gr.Textbox(label="HuggingFace Repo (e.g. strangerzonehf/Flux-Super-Realism-LoRA)")
            weights = gr.Textbox(label="Weights Filename (e.g. super-realism.safetensors)")
            trigger_word = gr.Textbox(label="Trigger Word")
            add_btn = gr.Button("Add LoRA")
            add_btn.click(add_custom_lora, inputs=[title, repo, weights, trigger_word], outputs=None)
    return demo

if __name__ == "__main__":
    gradio_ui().launch() 