import os
from PIL import Image
from utils import *
import torch
from diffusers import StableDiffusionPipeline

device = "cuda:0"

pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    torch_dtype=torch.float16
)
print("[INFO] Successfully loaded Stable Diffusion!")

# LoRA 權重路徑
lora_path = "/work_b/Sean/runs/dreambooth_cat"
pipe.load_lora_weights(
    lora_path,
    weight_name="pytorch_lora_weights.safetensors",
    local_files_only=True
)
print("[INFO] Successfully loaded LoRA weights!")

pipe = pipe.to(device)

# Inference
prompt = "a man with sunglasses"
seed = 10
seed_everything(seed)

image = pipe(
    prompt, 
    num_inference_steps=30, 
    guidance_scale=7.5
).images[0]

image.show()
