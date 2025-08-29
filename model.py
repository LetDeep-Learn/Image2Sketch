# model.py - corrected: adapters created on same device & dtype as base linear
import math
import torch
import torch.nn as nn
from diffusers import StableDiffusionPipeline, DDPMScheduler
from typing import Dict, Tuple

def load_base_pipeline(model_id: str, device="cuda", dtype=torch.float16, use_auth_token=None):
    load_kwargs = {}
    try:
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype, **load_kwargs)
    except TypeError:
        pipe = StableDiffusionPipeline.from_pretrained(model_id)
        if dtype is not None:
            pipe.to(dtype)
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

class LoRALinear(nn.Module):
    """
    LoRA wrapper that ensures adapter params are float32 and on the same device
    as the wrapped base linear. The LoRA output is cast to the base output dtype
    before addition to keep forward behavior consistent.
    """
    def __init__(self, base_linear: nn.Linear, r: int = 4, alpha: int = 16, dropout: float = 0.0):
        super().__init__()
        assert isinstance(base_linear, nn.Linear), "base_linear must be nn.Linear"
        self.base = base_linear
        # freeze base weights
        for p in self.base.parameters():
            p.requires_grad = False

        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features

        self.r = r
        self.alpha = alpha
        self.scaling = (alpha / r) if (r and r > 0) else 0.0
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0.0 else nn.Identity()

        # determine base device and base dtype
        try:
            base_param = next(self.base.parameters())
            base_device = base_param.device
            base_dtype = base_param.dtype
        except StopIteration:
            base_device = torch.device("cpu")
            base_dtype = torch.float32

        # Use adapters in float32 for stable gradients/unscale behavior
        adapter_dtype = torch.float32

        if self.r and self.r > 0:
            # create adapters explicitly on base device but with float32 dtype
            self.lora_down = nn.Linear(self.in_features, self.r, bias=False).to(device=base_device, dtype=adapter_dtype)
            self.lora_up = nn.Linear(self.r, self.out_features, bias=False).to(device=base_device, dtype=adapter_dtype)
            # init: up zeros so initial behaviour equals base
            nn.init.zeros_(self.lora_up.weight)
            nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        else:
            self.lora_down = None
            self.lora_up = None

    def forward(self, x: torch.Tensor):
        # Ensure input dtype matches base so base linear receives expected dtype
        base_param_dtype = next(self.base.parameters()).dtype
        if x.dtype != base_param_dtype:
            x_casted_for_base = x.to(base_param_dtype)
        else:
            x_casted_for_base = x

        base_out = self.base(x_casted_for_base)

        if self.r and self.r > 0:
            # compute LoRA path in float32 (adapters are float32)
            l_in = x.to(self.lora_down.weight.dtype) if x.dtype != self.lora_down.weight.dtype else x
            l = self.lora_up(self.lora_down(self.dropout(l_in))) * self.scaling
            # cast LoRA output to base_out dtype before adding (usually fp16 if model is in fp16)
            if l.dtype != base_out.dtype:
                l = l.to(base_out.dtype)
            return base_out + l
        else:
            return base_out

def inject_lora_to_unet_by_replacing_linears(unet: nn.Module, r: int = 4, alpha: int = 16, dropout: float = 0.0) -> Dict[str, LoRALinear]:
    candidate_attr_names = [
        "to_q", "to_k", "to_v", "to_out",
        "proj_attn", "to_q_proj", "to_k_proj", "to_v_proj", "to_out_proj",
        "q_proj", "k_proj", "v_proj", "out_proj"
    ]
    lora_modules = {}
    for module_name, module in unet.named_modules():
        for attr in candidate_attr_names:
            if hasattr(module, attr):
                orig = getattr(module, attr)
                if isinstance(orig, LoRALinear):
                    continue
                if isinstance(orig, nn.Linear):
                    lora_layer = LoRALinear(orig, r=r, alpha=alpha, dropout=dropout)
                    setattr(module, attr, lora_layer)
                    lora_modules[f"{module_name}.{attr}"] = lora_layer

    if len(lora_modules) == 0:
        for module_name, module in unet.named_modules():
            child_modules = list(module._modules.items())
            for child_name, child in child_modules:
                if isinstance(child, LoRALinear):
                    continue
                if isinstance(child, nn.Linear):
                    lora_layer = LoRALinear(child, r=r, alpha=alpha, dropout=dropout)
                    module._modules[child_name] = lora_layer
                    lora_modules[f"{module_name}.{child_name}"] = lora_layer
    return lora_modules

def collect_lora_parameters_from_unet(unet: nn.Module) -> Tuple[list, list]:
    lora_params = []
    lora_param_names = []
    for name, module in unet.named_modules():
        if isinstance(module, LoRALinear):
            if module.lora_down is not None:
                for p_name, p in module.lora_down.named_parameters():
                    lora_params.append(p)
                    lora_param_names.append(f"{name}.lora_down.{p_name}")
            if module.lora_up is not None:
                for p_name, p in module.lora_up.named_parameters():
                    lora_params.append(p)
                    lora_param_names.append(f"{name}.lora_up.{p_name}")
    return lora_params, lora_param_names

def save_lora_state(unet: nn.Module, path: str):
    sd = {}
    for name, module in unet.named_modules():
        if isinstance(module, LoRALinear):
            entry = {}
            if module.lora_down is not None:
                entry["lora_down"] = {k: v.cpu() for k, v in module.lora_down.state_dict().items()}
            if module.lora_up is not None:
                entry["lora_up"] = {k: v.cpu() for k, v in module.lora_up.state_dict().items()}
            sd[name] = entry
    torch.save(sd, path)

def load_lora_state_to_unet(unet: nn.Module, path: str, map_location="cpu"):
    sd = torch.load(path, map_location=map_location)
    missing = []
    loaded = []
    for saved_name, entry in sd.items():
        found = False
        for name, module in unet.named_modules():
            if name == saved_name and isinstance(module, LoRALinear):
                if "lora_down" in entry and module.lora_down is not None:
                    module.lora_down.load_state_dict(entry["lora_down"])
                if "lora_up" in entry and module.lora_up is not None:
                    module.lora_up.load_state_dict(entry["lora_up"])
                found = True
                loaded.append(saved_name)
                break
        if not found:
            missing.append(saved_name)
    if missing:
        print("Warning: some saved LoRA modules were not found in the current UNet:", missing)
    return loaded
