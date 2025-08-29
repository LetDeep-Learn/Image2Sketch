# inference.py
import sys
from PIL import Image
import torch
from diffusers import StableDiffusionImg2ImgPipeline
from config import *
from model import load_base_pipeline

def load_pipeline_with_lora(lora_dir, device="cuda"):
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float16).to(device)
    pipe.safety_checker = None
    pipe.enable_attention_slicing()
    pipe.enable_vae_tiling()
    # load LoRA weights (folder saved by AttnProcsLayers)
    pipe.load_lora_weights(lora_dir)
    return pipe

def photo_to_sketch(in_path, out_path, lora_dir=OUTPUT_DIR, prompt=INFER_PROMPT, strength=INFER_STRENGTH):
    pipe = load_pipeline_with_lora(lora_dir)
    im = Image.open(in_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    result = pipe(prompt=prompt, image=im, strength=strength, guidance_scale=INFER_GUIDANCE, num_inference_steps=INFER_STEPS)
    out = result.images[0]
    out.save(out_path)
    print("Saved", out_path)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python inference.py input.jpg output.png [lora_dir]")
    else:
        inp, out = sys.argv[1], sys.argv[2]
        lora_dir = sys.argv[3] if len(sys.argv) > 3 else OUTPUT_DIR
        photo_to_sketch(inp, out, lora_dir)
