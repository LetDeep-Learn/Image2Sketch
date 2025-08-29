import os
from dataclasses import dataclass

@dataclass
class Config:
    MODEL_ID = "runwayml/stable-diffusion-v1-5"
    HF_TOKEN = None

    DATA_DIR = "data"
    CAPTIONS_TXT = "pencil sketch, clean line art, black and white, no shading, high contrast"
    OUTPUT_DIR = "/content/drive/MyDrive/image2sketch/checkpoints"

    DEVICE = "cuda"
    IMG_SIZE = 512
    BATCH_SIZE = 2
    LR = 1e-4
    MAX_STEPS = 8000
    WARMUP_STEPS = 50
    CHECKPOINT_EVERY = 500
    MIXED_PRECISION = "fp16"
    GRAD_ACCUMULATE = 1
    SEED = 42

    LORA_RANK = 32   ## 4 if not work
    LORA_ALPHA = 64  ## 16
    LORA_DROPOUT = 0.1  ## 0.0

    SCHEDULER = "ddpm"
    SAVE_FINAL = True

cfg = Config()
