import sys
import subprocess

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Install dependencies if running in Colab
try:
    import google.colab
    install('gradio')
    install('diffusers')
    install('torch')
    install('transformers')
    install('accelerate')
    install('safetensors')
    install('Pillow')
    install('huggingface_hub')
except ImportError:
    pass

import torch
import gradio as gr
from diffusers import DiffusionPipeline, AutoencoderTiny, AutoencoderKL, AutoPipelineForImage2Image
from huggingface_hub import hf_hub_download
from PIL import Image

# LoRA DLCs (short list for demo, add more as needed)
loras = [
    {"title": "Super Realism", "repo": "strangerzonehf/Flux-Super-Realism-LoRA", "weights": "super-realism.safetensors", "image": "https://huggingface.co/strangerzonehf/Flux-Super-Realism-LoRA/resolve/main/images/1.png", "trigger_word": "Super Realism"},
    {"title": "Dalle Mix", "repo": "prithivMLmods/Flux-Dalle-Mix-LoRA", "weights": "dalle-mix.safetensors", "image": "https://huggingface.co/prithivMLmods/Flux-Dalle-Mix-LoRA/resolve/main/images/D3.png", "trigger_word": "dalle-mix"},
    {"title": "Ghibli Flux", "repo": "strangerzonehf/Ghibli-Flux-Cartoon-LoRA", "weights": "Ghibili-Cartoon-Art.safetensors", "image": "https://huggingface.co/strangerzonehf/Ghibli-Flux-Cartoon-LoRA/resolve/main/images/3333.png", "trigger_word": "Ghibli Art"},
    {"title": "Sketch_Smudge", "repo": "strangerzonehf/Flux-Sketch-Smudge-LoRA", "weights": "Sketch-Smudge.safetensors", "image": "https://huggingface.co/strangerzonehf/Flux-Sketch-Smudge-LoRA/resolve/main/images/5.png", "trigger_word": "Sketch Smudge"},
    {"title": "Animeo Mix", "repo": "strangerzonehf/Flux-Animeo-v1-LoRA", "weights": "Animeo.safetensors", "image": "https://huggingface.co/strangerzonehf/Flux-Animeo-v1-LoRA/resolve/main/images/A4.png", "trigger_word": "Animeo"},
    {"title": "Animex Mix", "repo": "strangerzonehf/Flux-Animex-v2-LoRA", "weights": "Animex.safetensors", "image": "https://huggingface.co/strangerzonehf/Flux-Animex-v2-LoRA/resolve/main/images/A33.png", "trigger_word": "Animex"},
]

# Model and VAE setup
base_model = "black-forest-labs/FLUX.1-dev"
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

taef1 = AutoencoderTiny.from_pretrained("madebyollin/taef1", torch_dtype=dtype).to(device)
good_vae = AutoencoderKL.from_pretrained(base_model, subfolder="vae", torch_dtype=dtype).to(device)
pipe = DiffusionPipeline.from_pretrained(base_model, torch_dtype=dtype, vae=taef1).to(device)
pipe_i2i = AutoPipelineForImage2Image.from_pretrained(
    base_model,
    vae=good_vae,
    transformer=pipe.transformer,
    text_encoder=pipe.text_encoder,
    tokenizer=pipe.tokenizer,
    text_encoder_2=pipe.text_encoder_2,
    tokenizer_2=pipe.tokenizer_2,
    torch_dtype=dtype
)

MAX_SEED = 2**32-1

def get_lora_gallery():
    return [[lora["image"], lora["title"]] for lora in loras]

def add_custom_lora(custom_repo, custom_weights, custom_image_url, custom_title, custom_trigger):
    loras.append({
        "title": custom_title or f"Custom LoRA {len(loras)+1}",
        "repo": custom_repo,
        "weights": custom_weights,
        "image": custom_image_url or "",
        "trigger_word": custom_trigger or ""
    })
    return get_lora_gallery(), gr.update(value="", visible=True), gr.update(value="", visible=True), gr.update(value="", visible=True), gr.update(value="", visible=True), gr.update(value="", visible=True)

def select_lora(evt: gr.SelectData):
    return evt.index

def generate_image(prompt, negative_prompt, steps, guidance_scale, width, height, seed, lora_idx, lora_scale):
    selected_lora = loras[lora_idx]
    lora_path = hf_hub_download(repo_id=selected_lora["repo"], filename=selected_lora["weights"])
    pipe.unload_lora_weights()
    pipe.load_lora_weights(lora_path, weight_name=None, adapter_name=None, scale=lora_scale)
    # Add trigger word if needed
    if selected_lora.get("trigger_word"):
        prompt = f"{selected_lora['trigger_word']} {prompt}"
    generator = torch.Generator(device=device).manual_seed(int(seed))
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
    gr.Markdown("# FLUX-LoRA-DLC (Advanced, Hugging Face Quality)")
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
            custom_trigger = gr.Textbox(label="Custom Trigger Word (optional)")
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
            lora_scale = gr.Slider(0, 3, value=1.0, label="LoRA Scale")
            generate_btn = gr.Button("Generate")
    lora_gallery.select(select_lora, None, lora_idx)
    generate_btn.click(
        generate_image,
        inputs=[prompt, negative_prompt, steps, guidance_scale, width, height, seed, lora_idx, lora_scale],
        outputs=output
    )
    add_custom_btn.click(
        add_custom_lora,
        inputs=[custom_repo, custom_weights, custom_image_url, custom_title, custom_trigger],
        outputs=[lora_gallery, custom_repo, custom_weights, custom_image_url, custom_title, custom_trigger]
    )

if __name__ == "__main__":
    demo.launch(share=True) 