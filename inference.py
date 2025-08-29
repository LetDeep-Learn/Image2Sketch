import os
import torch
from model import load_base_pipeline
from config import cfg

def load_lora_state_to_unet(unet, lora_path):
    sd = torch.load(lora_path, map_location="cpu")
    missing = []
    for name, state in sd.items():
        found = False
        for module_name, module in unet.named_modules():
            if module_name == name:
                try:
                    module.load_state_dict(state, strict=False)
                    found = True
                    break
                except Exception:
                    continue
        if not found:
            missing.append(name)
    if len(missing) > 0:
        print("Warning: these LoRA modules were not matched on the UNet:", missing)

def generate(prompt, lora_path, out_path="out.png", num_inference_steps=50, guidance_scale=7.5, height=512, width=512):
    dtype = torch.float16 if cfg.MIXED_PRECISION=="fp16" else torch.float32
    pipe = load_base_pipeline(cfg.MODEL_ID, device=cfg.DEVICE, dtype=dtype, use_auth_token=cfg.HF_TOKEN)
    unet = pipe.unet
    load_lora_state_to_unet(unet, lora_path)
    pipe.unet = unet
    generator = torch.Generator(device=cfg.DEVICE).manual_seed(cfg.SEED)
    image = pipe(prompt=prompt, height=height, width=width, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, generator=generator).images[0]
    image.save(out_path)
    return out_path

if __name__ == "__main__":
    demo_prompt = "A fantasy portrait of a robotic fox, detailed, cinematic lighting"
    demo_lora = os.path.join(cfg.OUTPUT_DIR, "lora_final.pt")
    if not os.path.isfile(demo_lora):
        raise FileNotFoundError("LoRA weights not found at " + demo_lora)
    out = generate(demo_prompt, demo_lora, out_path="demo_out.png")
    print("Image saved to", out)
