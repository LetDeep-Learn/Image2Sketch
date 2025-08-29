# model.py
import torch
from diffusers import StableDiffusionPipeline, DDPMScheduler
from diffusers.models.attention_processor import LoRAAttnProcessor
from diffusers.loaders import AttnProcsLayers

def load_base_pipeline(model_id, device="cuda", dtype=torch.float16):
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
    )
    pipe = pipe.to(device)
    pipe.safety_checker = None
    # Use scheduler config friendly for training
    pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    # memory helpers
    pipe.enable_vae_tiling()
    pipe.enable_attention_slicing()
    return pipe

def inject_lora_into_unet(pipe, rank=8):
    """
    Attach LoRA processors to every attention module with set_processor
    Returns the AttnProcsLayers wrapper which can be saved/loaded.
    """
    # iterate modules and attach LoRAAttnProcessor where possible
    for name, module in pipe.unet.named_modules():
        if hasattr(module, "set_processor"):
            hidden_size = getattr(module, "to_q").in_features
            cross_attn_dim = getattr(pipe.unet.config, "cross_attention_dim", None)
            proc = LoRAAttnProcessor(rank=rank)
            module.set_processor(proc)

    attn_procs = pipe.unet.attn_processors
    lora_layers = AttnProcsLayers(attn_procs)  # wrapper object
    return lora_layers

    
