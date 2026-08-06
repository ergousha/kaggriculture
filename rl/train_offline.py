"""Offline Training Pipeline for Kaggriculture.

Implements Behavior Cloning (BC) on the Macro-Actions dataset
exported by dataset_builder.py.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from rl.architecture import KaggriculturePolicy


def train_offline_bc(dataset_path: str, epochs: int = 5, batch_size: int = 256):
    """Trains a Behavior Cloning model on the provided dataset."""
    print(f"Loading dataset from {dataset_path}...")
    try:
        data = np.load(dataset_path)
        planes = torch.tensor(data["planes"], dtype=torch.float32)
        globals_vec = torch.tensor(data["globals"], dtype=torch.float32)
        actions = torch.tensor(data["actions"], dtype=torch.float32)
        print(f"Dataset loaded. {len(planes)} samples.")
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    dataset = TensorDataset(globals_vec, planes, actions)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print("Initializing ObservationEncoder and MacroActionDecoder...")
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    model = KaggriculturePolicy().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    print("Starting Behavior Cloning training loop...")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for batch_globals, batch_planes, batch_actions in dataloader:
            batch_globals = batch_globals.to(device)
            batch_planes = batch_planes.to(device)
            batch_actions = batch_actions.to(device)

            optimizer.zero_grad()

            logits = model(batch_globals, batch_planes)
            loss = criterion(logits, batch_actions)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}")

    import os

    os.makedirs("logs", exist_ok=True)
    save_path = "logs/bc_model.pt"
    torch.save(model.state_dict(), save_path)
    print(f"Training complete. Model saved to {save_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="logs/offline_rl_dataset.npz")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    train_offline_bc(args.dataset, args.epochs, args.batch_size)
