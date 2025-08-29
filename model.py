# model.py
import torch
from diffusers import StableDiffusionPipeline, DDPMScheduler
from diffusers.models.attention_processor import LoRAAttnProcessor2_0
from diffusers.loaders import AttnProcsLayers

def load_base_pipeline(model_id, device="cuda", dtype=torch.float16):
    """
    Load Stable Diffusion pipeline with scheduler set for training.
    Compatible with latest diffusers.
    """
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        safety_checker=None  # disable NSFW filter for training
    )
    pipe = pipe.to(device)

    # Use DDPM scheduler for training
    pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

    # Memory helpers
    pipe.enable_vae_tiling()
    pipe.enable_attention_slicing()
    return pipe


def inject_lora_into_unet(unet, rank=4):
    """
    Inject LoRA processors into UNet attention layers (latest diffusers API).
    Returns AttnProcsLayers containing LoRA weights.
    """
    attn_procs = {}

    for name, _ in unet.attn_processors.items():
        # cross-attn or self-attn
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim

        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name.split(".")[1])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name.split(".")[1])
            hidden_size = unet.config.block_out_channels[block_id]
        else:
            raise ValueError(f"Cannot determine hidden size for {name}")

        attn_procs[name] = LoRAAttnProcessor2_0(
            hidden_size=hidden_size,
            cross_attention_dim=cross_attention_dim,
            rank=rank
        )

    unet.set_attn_processor(attn_procs)
    return AttnProcsLayers(unet.attn_processors)
