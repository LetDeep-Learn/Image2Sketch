# train.py
import os, math, argparse
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from config import *
from dataset import SketchDataset
from model import load_base_pipeline, inject_lora_into_unet

from diffusers.loaders import AttnProcsLayers
from transformers import get_cosine_schedule_with_warmup

def main():
    # args override via env or CLI if wanted
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Device & pipeline
    device = DEVICE
    dtype = torch.float16 if MIXED_PRECISION == "fp16" else torch.bfloat16
    pipe = load_base_pipeline(MODEL_ID, device=device, dtype=dtype)

    # Freeze base networks
    pipe.unet.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.vae.requires_grad_(False)

    # Inject LoRA processors and collect trainable params
    lora_layers = inject_lora_into_unet(pipe, rank=LORA_RANK)
    trainable_params = list(lora_layers.parameters())
    if len(trainable_params) == 0:
        raise RuntimeError("No LoRA params found! Did injection fail?")

    # Optimizer & scheduler
    optimizer = torch.optim.AdamW(trainable_params, lr=LR)
    total_steps = MAX_STEPS
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=total_steps)

    # Dataset & Dataloader
    ds = SketchDataset(DATA_DIR, size=IMG_SIZE)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, drop_last=True)

    # Training loop
    pipe.unet.train()
    scaler = torch.cuda.amp.GradScaler(enabled=(MIXED_PRECISION=="fp16"))
    step = 0
    pbar = tqdm(total=total_steps)
    while step < total_steps:
        for batch in loader:
            step += 1
            pbar.update(1)

            pixel_values = batch["pixel_values"].to(device, dtype=pipe.vae.dtype)
            captions = [TRAINING_CAPTION] * pixel_values.shape[0]

            # encode to latents with VAE (no grad)
            with torch.no_grad():
                latents = pipe.vae.encode(pixel_values).latent_dist.sample()
                latents = latents * pipe.vae.config.scaling_factor

            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, pipe.scheduler.config.num_train_timesteps, (latents.shape[0],), device=latents.device).long()
            noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)

            # text embeddings
            text_inputs = pipe.tokenizer(captions, padding="max_length", truncation=True, max_length=pipe.tokenizer.model_max_length, return_tensors="pt")
            text_embeds = pipe.text_encoder(text_inputs.input_ids.to(device))[0]

            # forward with LoRA active (only LoRA params are trainable)
            with torch.cuda.amp.autocast(enabled=(MIXED_PRECISION=="fp16")):
                noise_pred = pipe.unet(noisy_latents, timesteps, encoder_hidden_states=text_embeds).sample
                loss = F.mse_loss(noise_pred.float(), noise.float())

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            if step % 10 == 0:
                pbar.set_description(f"loss: {loss.item():.4f}")

            if step % CHECKPOINT_EVERY == 0 or step == total_steps:
                outdir = os.path.join(OUTPUT_DIR, f"checkpoint-step-{step}")
                os.makedirs(outdir, exist_ok=True)
                lora_layers.save_pretrained(outdir)
                print(f"[SAVE] LoRA saved to {outdir}")

            if step >= total_steps:
                break

    # final save
    lora_layers.save_pretrained(OUTPUT_DIR)
    print("Training finished. LoRA saved to", OUTPUT_DIR)

if __name__ == "__main__":
    main()
