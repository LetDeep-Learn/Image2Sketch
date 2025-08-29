# # inference.py
# import torch
# from diffusers import StableDiffusionPipeline
# from model import load_lora_state_to_unet

# # -----------------------------
# # Config
# # -----------------------------
# BASE_MODEL = "runwayml/stable-diffusion-v1-5"   # Base SD model
# LORA_CKPT = "/content/drive/MyDrive/image2sketch/checkpoints/lora_checkpoint_step_450.pt"  # Change to your latest checkpoint
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# # -----------------------------
# # Load Base Model
# # -----------------------------
# print(f"Loading base model: {BASE_MODEL}")
# pipe = StableDiffusionPipeline.from_pretrained(
#     BASE_MODEL,
#     torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
# )
# pipe = pipe.to(DEVICE)

# # -----------------------------
# # Load LoRA Weights
# # -----------------------------
# print(f"Loading LoRA weights from: {LORA_CKPT}")
# load_lora_state_to_unet(pipe.unet, LORA_CKPT, map_location=DEVICE)
# print("✅ LoRA successfully loaded into UNet.")

# # -----------------------------
# # Run Inference (Image → Sketch)
# # -----------------------------
# # Instead of prompt-based generation, we pass an input image
# from PIL import Image

# INPUT_IMAGE = "data/images/img3.png"   # your source image
# OUTPUT_IMAGE = "data/images/output_sketch.png"

# # load image
# init_image = Image.open(INPUT_IMAGE).convert("RGB").resize((512, 512))

# # Generate sketch
# with torch.autocast(DEVICE if DEVICE == "cuda" else "cpu"):
#     result = pipe(
#         prompt="",          # we don’t need a text prompt
#         image=init_image,   # direct image → sketch
#         strength=1.0,
#         guidance_scale=7.5
#     )

# # Save result
# result.images[0].save(OUTPUT_IMAGE)
# print(f"✅ Sketch saved at {OUTPUT_IMAGE}")





import os
import torch
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline
from peft import PeftModel
from model import load_base_pipeline  # your helper to load base model

# ---------------- CONFIG ----------------
BASE_MODEL = "runwayml/stable-diffusion-v1-5"      # your base Stable Diffusion
LORA_PATH = "/content/drive/MyDrive/image2sketch/checkpoints/lora_checkpoint_step_450.pt"  # latest LoRA checkpoint
INPUT_IMAGE = "data/images/img3.png"   # your source image
OUTPUT_IMAGE = "data/images/output_sketch.png"               # save result
DEVICE = "cuda"
IMG_SIZE = (512, 512)                             # resize input image if needed
PROMPT = "sketch of the given image"
STRENGTH = 0.7                                    # 0 = preserve input, 1 = redraw completely
GUIDANCE_SCALE = 7.5                              # classifier-free guidance
NUM_INFERENCE_STEPS = 50
# ----------------------------------------

# 1️⃣ Load pipeline
print("[INFO] Loading base pipeline...")
pipe = load_base_pipeline(BASE_MODEL, device=DEVICE, dtype=torch.float16)  # your function
unet = pipe.unet

# 2️⃣ Load LoRA weights into UNet
print("[INFO] Loading LoRA weights...")
from model import load_lora_state_to_unet  # your custom loader
load_lora_state_to_unet(unet, LORA_PATH, map_location=DEVICE)
print("✅ LoRA successfully loaded into UNet.")

# 3️⃣ Switch to image-to-image pipeline
img2img_pipe = StableDiffusionImg2ImgPipeline(
    vae=pipe.vae,
    text_encoder=pipe.text_encoder,
    tokenizer=pipe.tokenizer,
    unet=unet,
    scheduler=pipe.scheduler,
    safety_checker=None,  # optional
    feature_extractor=pipe.feature_extractor
)
img2img_pipe = img2img_pipe.to(DEVICE)
img2img_pipe.enable_attention_slicing()  # save VRAM

# 4️⃣ Load and preprocess input image
init_image = Image.open(INPUT_IMAGE).convert("RGB")
init_image = init_image.resize(IMG_SIZE)

# 5️⃣ Run inference
print("[INFO] Generating sketch...")
output = img2img_pipe(
    prompt=PROMPT,
    image=init_image,
    strength=STRENGTH,
    guidance_scale=GUIDANCE_SCALE,
    num_inference_steps=NUM_INFERENCE_STEPS
)
sketch = output.images[0]

# 6️⃣ Save output
sketch.save(OUTPUT_IMAGE)
print(f"[INFO] Saved sketch image at {OUTPUT_IMAGE}")
