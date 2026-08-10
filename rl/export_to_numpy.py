"""Export PyTorch Model to Numpy for Kaggle Submission.

Kaggle environment submissions have a 100 MiB limit and restrict imports to the standard library + numpy.
This script loads the trained PyTorch Behavior Cloning model and extracts the weights/biases
into a lightweight `.npz` file that can be loaded natively by numpy in `main.py`.
"""

import os

import numpy as np
import torch

from rl.architecture import KaggriculturePolicyFullRL


def export_weights_to_numpy(model_path: str, output_path: str):
    print(f"Loading PyTorch model from {model_path}...")

    model = KaggriculturePolicyFullRL()

    try:
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    model.eval()

    print("Extracting weights and biases...")
    np_weights = {}

    for name, param in model.named_parameters():
        np_weights[name] = param.detach().cpu().numpy()

    if os.path.dirname(output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez_compressed(output_path, **np_weights)  # type: ignore
    print(f"Exported {len(np_weights)} parameter tensors to {output_path}")
    print(f"File size: {os.path.getsize(output_path) / 1024:.2f} KB (well within 100 MiB limit)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="logs/bc_model.pt")
    parser.add_argument("--output", default="logs/bc_weights.npz")
    args = parser.parse_args()

    export_weights_to_numpy(args.model, args.output)
