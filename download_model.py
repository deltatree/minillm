#!/usr/bin/env python3
"""
Download Gemma 3 1B GGUF model from Hugging Face.
Supports Q4_K_M and Q8_0 quantizations.
"""

import argparse
import os
from huggingface_hub import hf_hub_download


# Available quantizations for Gemma 3 1B
MODELS = {
    "q4": {
        "repo": "bartowski/google_gemma-3-1b-it-GGUF",
        "filename": "google_gemma-3-1b-it-Q4_K_M.gguf",
        "description": "4-bit quantization (~0.7 GB) – fast, low memory",
    },
    "q8": {
        "repo": "bartowski/google_gemma-3-1b-it-GGUF",
        "filename": "google_gemma-3-1b-it-Q8_0.gguf",
        "description": "8-bit quantization (~1.2 GB) – better quality",
    },
}


def download(quant: str, models_dir: str = "models") -> str:
    """Download the chosen GGUF model and return its local path."""
    info = MODELS[quant]
    os.makedirs(models_dir, exist_ok=True)

    print(f"Downloading {info['filename']} …")
    print(f"  Repo : {info['repo']}")
    print(f"  Desc : {info['description']}")

    path = hf_hub_download(
        repo_id=info["repo"],
        filename=info["filename"],
        local_dir=models_dir,
    )
    print(f"Model saved to: {path}")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Gemma 3 1B GGUF model")
    parser.add_argument(
        "--quant",
        choices=["q4", "q8"],
        default="q4",
        help="Quantization variant (default: q4)",
    )
    parser.add_argument(
        "--models-dir",
        default="models",
        help="Directory to store the model (default: models/)",
    )
    args = parser.parse_args()
    download(args.quant, args.models_dir)
