# model.py
import torch
import torch.nn as nn
from diffusers import StableDiffusionPipeline, DDPMScheduler


# -------------------------------
# Simple LoRA Linear wrapper
# -------------------------------
class LoRALinear(nn.Module):
    def __init__(self, orig_linear, rank=4, alpha=1.0):
        super().__init__()
        self.orig_linear = orig_linear
        self.rank = rank
        self.alpha = alpha

        self.lora_down = nn.Linear(orig_linear.in_features, rank, bias=False)
        self.lora_up = nn.Linear(rank, orig_linear.out_features, bias=False)

        # init
        nn.init.kaiming_uniform_(self.lora_down.weight, a=5**0.5)
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, x):
        return self.orig_linear(x) + self.lora_up(self.lora_down(x)) * (self.alpha / self.rank)


# -------------------------------
# Load Base SD1.5 Pipeline
# -------------------------------
def load_base_pipeline(model_id="runwayml/stable-diffusion-v1-5", device="cuda", dtype=torch.float16):
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
    )
    pipe = pipe.to(device)
    pipe.safety_checker = None

    # use training-friendly scheduler
    pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

    # memory helpers
    if hasattr(pipe, "enable_vae_tiling"):
        pipe.enable_vae_tiling()
    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()

    return pipe


# -------------------------------
# Inject LoRA into UNet (diffusers 0.14.0 compatible)
# -------------------------------
def inject_lora_into_unet(unet, rank=4, alpha=1.0):
    """
    Replace Linear layers in attention with LoRA-wrapped layers.
    Returns nn.ModuleList of trainable LoRA layers.
    """
    lora_layers = []

    for name, module in unet.named_modules():
        if isinstance(module, nn.Linear):
            parent = unet
            *path, last = name.split(".")
            for p in path:
                parent = getattr(parent, p)

            orig_linear = getattr(parent, last)
            lora_linear = LoRALinear(orig_linear, rank=rank, alpha=alpha)
            setattr(parent, last, lora_linear)
            lora_layers.append(lora_linear)

    return nn.ModuleList(lora_layers)
