import torch
from diffusers import StableDiffusionPipeline, DDPMScheduler

def load_base_pipeline(model_id, device="cuda", dtype=torch.float16, use_auth_token=None):
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        safety_checker=None,
        use_auth_token=use_auth_token,
    )
    try:
        pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    except Exception:
        pass

    pipe.to(device)
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pass
    try:
        pipe.enable_attention_slicing()
    except Exception:
        pass
    try:
        pipe.vae.enable_vae_slicing()
    except Exception:
        pass
    return pipe

def get_lora_attn_processor_class():
    try:
        from diffusers.models.attention_processor import LoRAAttnProcessor
        return LoRAAttnProcessor
    except Exception as e:
        raise ImportError("LoRAAttnProcessor not found in diffusers. Please install a compatible diffusers version (e.g., 0.20.x).") from e

def inject_lora_to_unet(unet, rank=4, alpha=16, dropout=0.0, device="cuda"):
    LoRAClass = get_lora_attn_processor_class()
    lora_state = {}
    for name, module in unet.named_modules():
        if "attn" in name or "attention" in name:
            try:
                hidden_size = None
                cross_attention_dim = None
                if hasattr(module, "to_q"):
                    hidden_size = getattr(module, "to_q").in_features
                elif hasattr(module, "query"):
                    hidden_size = getattr(module, "query").in_features
                else:
                    hidden_size = getattr(unet.config, "sample_size", None) or (getattr(unet.config, "block_out_channels", [-1])[-1])
                proc = LoRAClass(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, rank=rank, alpha=alpha, dropout=dropout)
                replaced = False
                for attr in ["processor", "attn1_processor", "attn2_processor"]:
                    if hasattr(module, attr):
                        setattr(module, attr, proc)
                        replaced = True
                if not replaced:
                    parts = name.split(".")
                    try:
                        cur = unet
                        for p in parts[:-1]:
                            cur = getattr(cur, p)
                        setattr(cur, parts[-1], proc)
                        replaced = True
                    except Exception:
                        replaced = False
                if replaced:
                    lora_state[name] = proc
            except Exception:
                continue
    return lora_state

def collect_lora_parameters(lora_state):
    params = []
    for name, module in lora_state.items():
        for p in module.parameters():
            if p.requires_grad:
                params.append(p)
    return params
