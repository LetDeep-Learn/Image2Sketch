# import os
# import torch
# from model import load_base_pipeline
# from config import cfg

# def load_lora_state_to_unet(unet, lora_path):
#     sd = torch.load(lora_path, map_location="cpu")
#     missing = []
#     for name, state in sd.items():
#         found = False
#         for module_name, module in unet.named_modules():
#             if module_name == name:
#                 try:
#                     module.load_state_dict(state, strict=False)
#                     found = True
#                     break
#                 except Exception:
#                     continue
#         if not found:
#             missing.append(name)
#     if len(missing) > 0:
#         print("Warning: these LoRA modules were not matched on the UNet:", missing)

# def generate(prompt, lora_path, out_path="out.png", num_inference_steps=50, guidance_scale=7.5, height=512, width=512):
#     dtype = torch.float16 if cfg.MIXED_PRECISION=="fp16" else torch.float32
#     pipe = load_base_pipeline(cfg.MODEL_ID, device=cfg.DEVICE, dtype=dtype, use_auth_token=cfg.HF_TOKEN)
#     unet = pipe.unet
#     load_lora_state_to_unet(unet, lora_path)
#     pipe.unet = unet
#     generator = torch.Generator(device=cfg.DEVICE).manual_seed(cfg.SEED)
#     image = pipe(prompt=prompt, height=height, width=width, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, generator=generator).images[0]
#     image.save(out_path)
#     return out_path

# if __name__ == "__main__":
#     demo_prompt = "A fantasy portrait of a robotic fox, detailed, cinematic lighting"
#     demo_lora = os.path.join(cfg.OUTPUT_DIR, "lora_final.pt")
#     if not os.path.isfile(demo_lora):
#         raise FileNotFoundError("LoRA weights not found at " + demo_lora)
#     out = generate(demo_prompt, demo_lora, out_path="demo_out.png")
#     print("Image saved to", out)




###################################################################################33
import torch
# from diffusers import StableDiffusionPipeline
from peft import PeftModel
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline

# Paths
BASE_MODEL = "runwayml/stable-diffusion-v1-5"   # or whichever base you used
LORA_PATH = "/content/drive/MyDrive/image2sketch/checkpoints"                      # your trained LoRA folder
OUTPUT_PATH = "generated.png"                 # output image file

# def load_pipeline():
#     print("[INFO] Loading base model...")
#     pipe = StableDiffusionPipeline.from_pretrained(
#         BASE_MODEL,
#         torch_dtype=torch.float16
#     ).to("cuda")

#     print("[INFO] Attaching LoRA weights...")
#     pipe.unet = PeftModel.from_pretrained(pipe.unet, LORA_PATH)

#     return pipe

# def run_inference(pipe):
#     print("[INFO] Generating image without caption...")
#     image = pipe(prompt="").images[0]  # empty prompt since no captions used
#     image.save(OUTPUT_PATH)
#     print(f"[INFO] Saved generated image at {OUTPUT_PATH}")

# if __name__ == "__main__":
#     pipe = load_pipeline()
#     run_inference(pipe)


# load input image
# inference.py
import torch
from diffusers import StableDiffusionPipeline
from model import load_lora_state_to_unet

# -----------------------------
# Config
# -----------------------------
BASE_MODEL = "runwayml/stable-diffusion-v1-5"   # Base SD model
LORA_CKPT = "/content/drive/MyDrive/image2sketch/checkpoints/lora_checkpoint_step_450.pt"  # Change to your latest checkpoint
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# Load Base Model
# -----------------------------
print(f"Loading base model: {BASE_MODEL}")
pipe = StableDiffusionPipeline.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
)
pipe = pipe.to(DEVICE)

# -----------------------------
# Load LoRA Weights
# -----------------------------
print(f"Loading LoRA weights from: {LORA_CKPT}")
load_lora_state_to_unet(pipe.unet, LORA_CKPT, map_location=DEVICE)
print("✅ LoRA successfully loaded into UNet.")

# -----------------------------
# Run Inference (Image → Sketch)
# -----------------------------
# Instead of prompt-based generation, we pass an input image
from PIL import Image

INPUT_IMAGE = "data/images/img3.png"   # your source image
OUTPUT_IMAGE = "/output_sketch.png"

# load image
init_image = Image.open(INPUT_IMAGE).convert("RGB").resize((512, 512))

# Generate sketch
with torch.autocast(DEVICE if DEVICE == "cuda" else "cpu"):
    result = pipe(
        prompt="",          # we don’t need a text prompt
        image=init_image,   # direct image → sketch
        strength=1.0,
        guidance_scale=7.5
    )

# Save result
result.images[0].save(OUTPUT_IMAGE)
print(f"✅ Sketch saved at {OUTPUT_IMAGE}")
