import os
import random
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch import amp
from torch.optim.lr_scheduler import LambdaLR
from config import cfg
from old_cap_dataset import ImageDataset, collate_fn
from model import (
    load_base_pipeline,
    inject_lora_to_unet_by_replacing_linears,
    collect_lora_parameters_from_unet,
)

# ---------------------------
# Utils
# ---------------------------
def seed_everything(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def save_checkpoint(unet, optimizer, scheduler, scaler, global_step, path):
    """Save LoRA + optimizer/lr/scaler state"""
    lora_state = {}
    for name, module in unet.named_modules():
        if hasattr(module, "lora_up") or hasattr(module, "lora_down"):
            entry = {}
            if getattr(module, "lora_down", None) is not None:
                entry["lora_down"] = {k: v.cpu() for k, v in module.lora_down.state_dict().items()}
            if getattr(module, "lora_up", None) is not None:
                entry["lora_up"] = {k: v.cpu() for k, v in module.lora_up.state_dict().items()}
            if entry:
                lora_state[name] = entry

    torch.save({
        "step": global_step,
        "lora": lora_state,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
    }, path)
    print(f"[INFO] Saved checkpoint at {path} (step {global_step})")

def load_checkpoint(unet, optimizer, scheduler, scaler, path, device):
    """Restore LoRA + optimizer/lr/scaler state"""
    ckpt = torch.load(path, map_location=device)

    # Restore LoRA weights
    for saved_name, entry in ckpt["lora"].items():
        for name, module in unet.named_modules():
            if name == saved_name:
                if getattr(module, "lora_down", None) is not None and "lora_down" in entry:
                    module.lora_down.load_state_dict(entry["lora_down"])
                if getattr(module, "lora_up", None) is not None and "lora_up" in entry:
                    module.lora_up.load_state_dict(entry["lora_up"])
                break

    # Restore training states
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    scaler.load_state_dict(ckpt["scaler"])
    step = ckpt["step"]

    print(f"[INFO] Resumed training from step {step} (loaded {path})")
    return step

# ---------------------------
# Training
# ---------------------------
def main():
    seed_everything(cfg.SEED)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    if cfg.MIXED_PRECISION == "fp16":
        dtype = torch.float16
    elif cfg.MIXED_PRECISION == "bf16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    pipe = load_base_pipeline(cfg.MODEL_ID, device=cfg.DEVICE, dtype=dtype, use_auth_token=cfg.HF_TOKEN)
    unet = pipe.unet
    vae = pipe.vae

    # Freeze everything except LoRA adapters
    for p in unet.parameters():
        p.requires_grad = False
    for p in vae.parameters():
        p.requires_grad = False

    inject_lora_to_unet_by_replacing_linears(
        unet, r=cfg.LORA_RANK, alpha=cfg.LORA_ALPHA, dropout=cfg.LORA_DROPOUT
    )
    trainable_params, trainable_names = collect_lora_parameters_from_unet(unet)
    print(f"[INFO] LoRA adapter tensors found: {len(trainable_params)}")
    print(f"[INFO] Example adapter param names: {trainable_names[:10]}")
    if len(trainable_params) == 0:
        raise RuntimeError("No LoRA parameters found to train. Injector likely failed.")

    optimizer = AdamW(trainable_params, lr=cfg.LR)

    def lr_lambda(step):
        if step < cfg.WARMUP_STEPS:
            return float(step) / max(1.0, cfg.WARMUP_STEPS)
        else:
            return max(0.0, float(cfg.MAX_STEPS - step) / float(max(1, cfg.MAX_STEPS - cfg.WARMUP_STEPS)))
    scheduler = LambdaLR(optimizer, lr_lambda)

    dataset = ImageDataset(cfg.DATA_DIR, img_size=cfg.IMG_SIZE)
    dataloader = DataLoader(dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, collate_fn=collate_fn, drop_last=True)

    scaler = amp.GradScaler(enabled=(cfg.MIXED_PRECISION != "no"))

    # ---------------------------
    # Resume logic
    # ---------------------------
    global_step = 0
    if hasattr(cfg, "RESUME_PATH") and cfg.RESUME_PATH and os.path.exists(cfg.RESUME_PATH):
        global_step = load_checkpoint(unet, optimizer, scheduler, scaler, cfg.RESUME_PATH, cfg.DEVICE)

    pbar = tqdm(total=cfg.MAX_STEPS, initial=global_step)
    try:
        pipe.scheduler.set_timesteps(1000)
    except Exception:
        pass

    # ---------------------------
    # Training loop
    # ---------------------------
    for epoch in range(100000):
        for batch in dataloader:
            if global_step >= cfg.MAX_STEPS:
                break

            pixel_values = batch["pixel_values"].to(cfg.DEVICE)

            # Encode with VAE
            with torch.no_grad():
                latents = vae.encode(
                    pixel_values.half() if cfg.MIXED_PRECISION == "fp16" else pixel_values
                ).latent_dist.sample() * vae.config.scaling_factor

            noise = torch.randn_like(latents)
            try:
                timesteps = torch.randint(
                    0, pipe.scheduler.config.num_train_timesteps, (latents.shape[0],), device=latents.device
                ).long()
            except Exception:
                timesteps = torch.randint(0, 1000, (latents.shape[0],), device=latents.device).long()
            noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)

            optimizer.zero_grad()
            with amp.autocast(device_type="cuda", enabled=(cfg.MIXED_PRECISION != "no")):
                batch_size = latents.shape[0]
                hidden_size = unet.config.cross_attention_dim
                text_embeddings = torch.zeros(
                    (batch_size, 1, hidden_size), device=cfg.DEVICE, dtype=latents.dtype
                )

                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states=text_embeddings).sample
                loss = torch.nn.functional.mse_loss(model_pred.float(), noise.float(), reduction="mean")
                loss_to_backprop = loss / cfg.GRAD_ACCUMULATE

            scaler.scale(loss_to_backprop).backward()
            if (global_step + 1) % cfg.GRAD_ACCUMULATE == 0:
                try:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                except ValueError:
                    print("[WARNING] scaler.unscale_ failed. Skipping grad clipping.")
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

            global_step += 1
            pbar.update(1)
            pbar.set_description(f"step {global_step} loss {loss.item():.4f}")

            if global_step % cfg.CHECKPOINT_EVERY == 0 or global_step >= cfg.MAX_STEPS:
                ckpt_path = os.path.join(cfg.OUTPUT_DIR, f"lora_checkpoint_step_{global_step}.pt")
                save_checkpoint(unet, optimizer, scheduler, scaler, global_step, ckpt_path)

            if global_step >= cfg.MAX_STEPS:
                break

        if global_step >= cfg.MAX_STEPS:
            break

    if cfg.SAVE_FINAL:
        final_path = os.path.join(cfg.OUTPUT_DIR, "lora_final.pt")
        save_checkpoint(unet, optimizer, scheduler, scaler, global_step, final_path)

    print("Training finished. LoRA saved to", cfg.OUTPUT_DIR)

if __name__ == "__main__":
    main()
