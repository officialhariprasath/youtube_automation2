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

def generate_image(prompt, negative_prompt="", num_inference_steps=30, guidance_scale=7.5):
    pipe = load_model()
    
    # Generate the image
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale
    ).images[0]
    
    return image

# Create Gradio interface
def create_interface():
    iface = gr.Interface(
        fn=generate_image,
        inputs=[
            gr.Textbox(label="Prompt", placeholder="Enter your prompt here..."),
            gr.Textbox(label="Negative Prompt", placeholder="Enter what you don't want in the image..."),
            gr.Slider(minimum=1, maximum=100, value=30, label="Number of Steps"),
            gr.Slider(minimum=1, maximum=20, value=7.5, label="Guidance Scale")
        ],
        outputs=gr.Image(label="Generated Image"),
        title="FLUX-LoRA-DLC Text to Image Generator",
        description="Generate images from text using Stable Diffusion"
    )
    return iface

if __name__ == "__main__":
    # Create and launch the interface
    interface = create_interface()
    interface.launch() 