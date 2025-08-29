we’re focusing on fine-tuning Stable Diffusion’s image-to-image pipeline (v1.5) with LoRA adapters so it can take in a photo and generate a sketch version of it.

Your plan is:

Base model → runwayml/stable-diffusion-v1-5 (about 4 GB).

Training method → LoRA (so training is faster, lighter).

Input at inference → just an image (not a prompt), since we want strict photo → sketch conversion.

Output → a sketch-style image (black-and-white / pencil-like), consistently across inputs.

Dataset → paired sketches or just a big set of sketches depending on whether we want strict mapping vs style transfer.

That’s the exact project setup we were mapping out


DIRECTORY STRUCTURE

image2sketch/
│
├── data/
│   ├── sketches/        # training sketches for fine-tuning
│   ├── images/          # photos for validation/inference testing
│
├── config.py            # configs & hyperparameters
├── model.py             # model loading + LoRA adapter setup
├── train.py             # training loop
├── inference.py         # run sketch generation on validation images
├── utils.py             # preprocessing, logging, saving
├── requirements.txt
├── README.md
