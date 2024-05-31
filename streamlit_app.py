import mediapy as media
import random
import sys
import torch
import streamlit as st
from diffusers import DiffusionPipeline, TCDScheduler
from huggingface_hub import hf_hub_download
from PIL import Image
import io

# Define the function to generate the image
def generate_image(prompt):
    num_inference_steps = 8
    base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    repo_name = "ByteDance/Hyper-SD"
    plural = "s" if num_inference_steps > 1 else ""
    ckpt_name = f"Hyper-SDXL-{num_inference_steps}step{plural}-lora.safetensors"
    device = "cpu"  # Change to CPU

    pipe = DiffusionPipeline.from_pretrained(base_model_id, torch_dtype=torch.float32).to(device)
    pipe.load_lora_weights(hf_hub_download(repo_name, ckpt_name))
    pipe.fuse_lora()
    pipe.scheduler = TCDScheduler.from_config(pipe.scheduler.config)

    seed = random.randint(0, sys.maxsize)
    eta = 0.5

    images = pipe(
        prompt=prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=0.0,
        eta=eta,
        generator=torch.Generator(device).manual_seed(seed),
    ).images

    return images[0], seed

st.title("Image Generation from Prompt")

prompt = st.text_input("Enter a prompt for image generation:")

if st.button("Generate Image"):
    if prompt:
        with st.spinner("Generating image..."):
            image, seed = generate_image(prompt)
            st.image(image, caption=f"Generated image for prompt: {prompt}")
            st.write(f"Prompt: {prompt}")
            st.write(f"Seed: {seed}")
    else:
        st.write("Please enter a prompt")
