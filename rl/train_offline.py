"""Offline Training Pipeline for Kaggriculture Full RL.

Implements Behavior Cloning (BC) on the Full Atomic Actions dataset
exported by dataset_builder.py.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from rl.architecture import KaggriculturePolicyFullRL


def train_offline_bc(dataset_path: str, epochs: int = 5, batch_size: int = 256):
    """Trains a Behavior Cloning model on the provided dataset."""
    print(f"Loading dataset from {dataset_path}...")
    try:
        data = np.load(dataset_path)
        planes = torch.tensor(data["planes"], dtype=torch.float32)
        globals_vec = torch.tensor(data["globals"], dtype=torch.float32)

        farmer_acts = torch.tensor(data["farmer_acts"], dtype=torch.long)
        hands_acts = torch.tensor(data["hands_acts"], dtype=torch.long)
        market_acts = torch.tensor(data["market_acts"], dtype=torch.float32)
        print(f"Dataset loaded. {len(planes)} samples.")
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    dataset = TensorDataset(globals_vec, planes, farmer_acts, hands_acts, market_acts)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print("Initializing Full RL Policy...")
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    model = KaggriculturePolicyFullRL().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    ce_loss = nn.CrossEntropyLoss()
    bce_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([50.0, 50.0]).to(device))
    mse_loss = nn.MSELoss()

    print("Starting Behavior Cloning training loop...")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_f_loss = 0.0
        total_h_loss = 0.0
        total_m_loss = 0.0

        for batch_globals, batch_planes, batch_f, batch_h, batch_m in dataloader:
            batch_globals = batch_globals.to(device)
            batch_planes = batch_planes.to(device)
            batch_f = batch_f.to(device)
            batch_h = batch_h.to(device)
            batch_m = batch_m.to(device)

            optimizer.zero_grad()

            f_logits, h_logits, m_preds = model(batch_globals, batch_planes)

            # Farmer categorical loss
            loss_f = ce_loss(f_logits, batch_f)

            # Hands spatial categorical loss
            loss_h = ce_loss(h_logits, batch_h)

            # Market loss (BCE for buy/sell flags, MSE for log qty)
            loss_m_flags = bce_loss(m_preds[..., :2], batch_m[..., :2])
            active_m_mask = batch_m[..., 2] > 0
            if active_m_mask.any():
                loss_m_qty = mse_loss(
                    m_preds[..., 2][active_m_mask], batch_m[..., 2][active_m_mask]
                )
            else:
                loss_m_qty = torch.tensor(0.0, device=device)
            loss_m = loss_m_flags + loss_m_qty

            loss = loss_f + loss_h + loss_m

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_f_loss += loss_f.item()
            total_h_loss += loss_h.item()
            total_m_loss += loss_m.item()

        avg_loss = total_loss / len(dataloader)
        avg_f = total_f_loss / len(dataloader)
        avg_h = total_h_loss / len(dataloader)
        avg_m = total_m_loss / len(dataloader)
        print(
            f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f} (F: {avg_f:.4f}, H: {avg_h:.4f}, M: {avg_m:.4f})"
        )

    import os

    os.makedirs("logs", exist_ok=True)
    save_path = "logs/bc_full_model.pt"
    torch.save(model.state_dict(), save_path)
    print(f"Training complete. Model saved to {save_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="logs/offline_full_rl_dataset.npz")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    train_offline_bc(args.dataset, args.epochs, args.batch_size)
