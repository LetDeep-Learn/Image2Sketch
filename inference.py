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
OUTPUT_IMAGE = "data/images/output_sketch.png"

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
