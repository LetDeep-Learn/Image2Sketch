# config.py
import os

# Model / HF
MODEL_ID = "runwayml/stable-diffusion-v1-5"

# Data
DATA_DIR = "data/sketches"      # in repo (read-only) or mount drive path
EVAL_PHOTOS_DIR = "data/images" # optional for evaluation

# Output / checkpoints
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/content/drive/MyDrive/sd_lora_sketch")  # default to Drive mount
CHECKPOINT_EVERY = 1000

# Training hyperparams
IMG_SIZE = 512
BATCH_SIZE = 1                  # T4 can be tight; set 1 or 2
GRAD_ACCUM_STEPS = 1
LR = 1e-4
MAX_STEPS = 5000
WARMUP_STEPS = 200
LORA_RANK = 8

# Precision & device
MIXED_PRECISION = "fp16"        # use "fp16" on Colab GPU
DEVICE = "cuda"

# Caption used during LoRA training (style prompt)
TRAINING_CAPTION = "pencil sketch, clean line art, black and white, no shading, high contrast"

# Inference defaults
INFER_PROMPT = "pencil sketch, clean line art, black and white"
INFER_STRENGTH = 0.55
INFER_GUIDANCE = 7.5
INFER_STEPS = 30
