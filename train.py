import os
import random
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import LambdaLR

from config import cfg
from dataset import ImageCaptionDataset, collate_fn
from model import load_base_pipeline, inject_lora_to_unet, collect_lora_parameters

def seed_everything(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def save_lora_state(lora_state, path):
    sd = {}
    for name, mod in lora_state.items():
        try:
            sd[name] = mod.state_dict()
        except Exception:
            sd[name] = {k: v for k, v in mod.__dict__.items() if hasattr(v, 'state_dict')}
    torch.save(sd, path)

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
    text_encoder = pipe.text_encoder
    tokenizer = pipe.tokenizer

    for p in unet.parameters():
        p.requires_grad = False
    for p in vae.parameters():
        p.requires_grad = False
    for p in text_encoder.parameters():
        p.requires_grad = False

    lora_state = inject_lora_to_unet(unet, rank=cfg.LORA_RANK, alpha=cfg.LORA_ALPHA, dropout=cfg.LORA_DROPOUT, device=cfg.DEVICE)
    trainable_params = collect_lora_parameters(lora_state)
    if len(trainable_params) == 0:
        raise RuntimeError("No LoRA parameters found to train. Check diffusers version or injection logic.")

    optimizer = AdamW(trainable_params, lr=cfg.LR)

    def lr_lambda(step):
        if step < cfg.WARMUP_STEPS:
            return float(step) / max(1.0, cfg.WARMUP_STEPS)
        else:
            return max(0.0, float(cfg.MAX_STEPS - step) / float(max(1, cfg.MAX_STEPS - cfg.WARMUP_STEPS)))
    scheduler = LambdaLR(optimizer, lr_lambda)

    dataset = ImageCaptionDataset(cfg.DATA_DIR, img_size=cfg.IMG_SIZE, captions_txt=cfg.CAPTIONS_TXT)
    dataloader = DataLoader(dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, collate_fn=collate_fn, drop_last=True)

    scaler = GradScaler(enabled=(cfg.MIXED_PRECISION != "no"))

    global_step = 0
    pbar = tqdm(total=cfg.MAX_STEPS)
    try:
        pipe.scheduler.set_timesteps(1000)
    except Exception:
        pass

    for epoch in range(100000):
        for batch in dataloader:
            if global_step >= cfg.MAX_STEPS:
                break

            pixel_values = batch["pixel_values"].to(cfg.DEVICE)
            captions = batch["captions"]

            tokenized = tokenizer(captions, padding="max_length", truncation=True, max_length=tokenizer.model_max_length, return_tensors="pt")
            tokenized = {k: v.to(cfg.DEVICE) for k,v in tokenized.items()}
            with torch.no_grad():
                text_embeddings = text_encoder(**tokenized).last_hidden_state

            with torch.no_grad():
                latents = vae.encode(pixel_values.half() if cfg.MIXED_PRECISION=="fp16" else pixel_values).latent_dist.sample() * vae.config.scaling_factor

            noise = torch.randn_like(latents)
            try:
                timesteps = torch.randint(0, pipe.scheduler.config.num_train_timesteps, (latents.shape[0],), device=latents.device).long()
            except Exception:
                timesteps = torch.randint(0, 1000, (latents.shape[0],), device=latents.device).long()
            noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)

            optimizer.zero_grad()
            with autocast(enabled=(cfg.MIXED_PRECISION!="no")):
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states=text_embeddings).sample
                loss = torch.nn.functional.mse_loss(model_pred.float(), noise.float(), reduction="mean")
                loss_to_backprop = loss / cfg.GRAD_ACCUMULATE

            scaler.scale(loss_to_backprop).backward()
            if (global_step + 1) % cfg.GRAD_ACCUMULATE == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

            global_step += 1
            pbar.update(1)
            pbar.set_description(f"step {global_step} loss {loss.item():.4f}")

            if global_step % cfg.CHECKPOINT_EVERY == 0 or global_step >= cfg.MAX_STEPS:
                ckpt_path = os.path.join(cfg.OUTPUT_DIR, f"lora_checkpoint_step_{global_step}.pt")
                save_lora_state(lora_state, ckpt_path)

            if global_step >= cfg.MAX_STEPS:
                break

        if global_step >= cfg.MAX_STEPS:
            break

    if cfg.SAVE_FINAL:
        final_path = os.path.join(cfg.OUTPUT_DIR, "lora_final.pt")
        save_lora_state(lora_state, final_path)

    print("Training finished. LoRA saved to", cfg.OUTPUT_DIR)

if __name__ == "__main__":
    main()
