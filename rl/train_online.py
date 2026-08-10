import glob
import os
import subprocess

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from rl.architecture import KaggriculturePolicyFullRL
from rl.dataset_builder import parse_online_replay

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")


def run_rollouts(episodes: int = 20) -> list[str]:
    """Runs local_arena to generate new replays and returns their paths."""
    # Clean up old replays from local arena
    old_replays = glob.glob(os.path.join(LOG_DIR, "match_run_*.json"))
    for f in old_replays:
        try:
            os.remove(f)
        except OSError:
            pass

    cmd = (
        f"uv run python local_arena.py --agent main.py "
        f"--opponent opponents/v0_0_7.py "
        f"--episodes {episodes} --save-replays {episodes}"
    )
    print(f"Running rollouts: {cmd}")
    subprocess.run(cmd, shell=True, check=True, cwd=PROJECT_ROOT)

    new_replays = glob.glob(os.path.join(LOG_DIR, "match_run_*.json"))
    return new_replays


def train_online_pg(iterations: int = 10, episodes_per_iter: int = 20, entropy_coef: float = 0.01):
    """Runs a Policy Gradient loop."""
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    model = KaggriculturePolicyFullRL().to(device)
    model_path = os.path.join(LOG_DIR, "bc_full_model.pt")
    if os.path.exists(model_path):
        print(f"Loading base model from {model_path}...")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print("Warning: No base model found! Training from scratch.")

    optimizer = optim.Adam(model.parameters(), lr=1e-5) # Lower LR for finetuning

    # We do NOT use reduction="mean" because we will multiply the raw losses by Advantage!
    ce_loss = nn.CrossEntropyLoss(reduction="none")
    bce_loss = nn.BCEWithLogitsLoss(reduction="none", pos_weight=torch.tensor([50.0, 50.0]).to(device))
    mse_loss = nn.MSELoss(reduction="none")

    for iter_idx in range(1, iterations + 1):
        print(f"\n=== Online RL Iteration {iter_idx}/{iterations} ===")
        replays = run_rollouts(episodes=episodes_per_iter)
        if not replays:
            print("Error: No replays found after rollouts.")
            continue

        # Parse replays and collect advantages
        parsed_data = []
        cash_list = []
        for r_path in replays:
            # We play as player 0 normally, but local_arena swaps seats (target_p = 0 or 1)
            # local_arena.py swap logic: swap is bool(i % 2).
            # To be safe, we just use target_p=0 for even indices, target_p=1 for odd.
            # We can extract the seed from the filename to determine swap.
            try:
                seed = int(r_path.split("_")[-1].split(".")[0])
                target_p = 1 if seed % 2 != 0 else 0
            except ValueError:
                target_p = 0

            parsed = parse_online_replay(r_path, target_p=target_p)
            if parsed is not None:
                me_cash, opp_cash, arrays = parsed
                parsed_data.append(arrays)
                cash_list.append(me_cash)

        if not parsed_data:
            print("No valid trajectories parsed.")
            continue

        # Compute Advantages
        cash_array = np.array(cash_list, dtype=np.float32)
        mean_cash = cash_array.mean()
        std_cash = cash_array.std() + 1e-8
        advantages = (cash_array - mean_cash) / std_cash
        print(f"Iteration Cash - Mean: ${mean_cash:.2f}, Std: {std_cash:.2f}, Max: ${cash_array.max():.2f}")

        # Update Model
        model.train()
        total_loss = 0.0

        for traj_idx, arrays in enumerate(parsed_data):
            adv = advantages[traj_idx]
            planes, globals_vec, f_acts, h_acts, m_acts = arrays

            # Convert to tensors
            planes_t = torch.tensor(planes, dtype=torch.float32, device=device)
            globals_t = torch.tensor(globals_vec, dtype=torch.float32, device=device)
            f_acts_t = torch.tensor(f_acts, dtype=torch.long, device=device)
            h_acts_t = torch.tensor(h_acts, dtype=torch.long, device=device)
            m_acts_t = torch.tensor(m_acts, dtype=torch.float32, device=device)

            optimizer.zero_grad()
            f_logits, h_logits, m_preds = model(globals_t, planes_t)

            # Compute step-wise losses
            loss_f = ce_loss(f_logits, f_acts_t)

            # h_logits: (N, 18, 10, 10). batch_h: (N, 10, 10)
            # ce_loss works directly on this shape
            loss_h = ce_loss(h_logits, h_acts_t)
            # mean over spatial dims for each step
            loss_h = loss_h.mean(dim=(1, 2))

            loss_m_flags = bce_loss(m_preds[..., :2], m_acts_t[..., :2]).mean(dim=(1, 2))

            # MSE loss only for active market actions
            active_m_mask = (m_acts_t[..., 2] > 0)
            loss_m_qty = torch.zeros_like(loss_m_flags)
            if active_m_mask.any():
                # This is tricky because we need it per-step.
                # Just compute standard MSE and mask it
                raw_mse = mse_loss(m_preds[..., 2], m_acts_t[..., 2])
                raw_mse = raw_mse * active_m_mask.float()
                loss_m_qty = raw_mse.sum(dim=1) / (active_m_mask.sum(dim=1) + 1e-8)

            loss_m = loss_m_flags + loss_m_qty

            # Total unweighted step loss
            step_losses = loss_f + loss_h + loss_m

            # Multiply by Advantage
            pg_loss = (step_losses * adv).mean()

            # Entropy Bonus
            dist_f = torch.distributions.Categorical(logits=f_logits)
            entropy_f = dist_f.entropy().mean()

            # h_logits is (N, 18, 10, 10), permute to (N, 10, 10, 18) for Categorical
            dist_h = torch.distributions.Categorical(logits=h_logits.permute(0, 2, 3, 1))
            entropy_h = dist_h.entropy().mean()

            # Market BCE Entropy
            p_m = torch.sigmoid(m_preds[..., :2])
            entropy_m = (-p_m * torch.log(p_m + 1e-8) - (1 - p_m) * torch.log(1 - p_m + 1e-8)).mean()

            total_entropy = entropy_f + entropy_h + entropy_m
            pg_loss = pg_loss - (entropy_coef * total_entropy)

            pg_loss.backward()

            # Gradient clipping to prevent instability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += pg_loss.item()

        print(f"Iter PG Loss: {total_loss / len(parsed_data):.4f}")

        # Save model and rebuild main.py
        torch.save(model.state_dict(), model_path)
        print("Rebuilding main.py with new weights...")
        subprocess.run("uv run python scripts/build_submission.py", shell=True, check=True, cwd=PROJECT_ROOT)

    print("\nOnline RL Training Complete!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    args = parser.parse_args()

    train_online_pg(args.iterations, args.episodes, args.entropy_coef)
