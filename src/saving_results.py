import hashlib
import json
import os
from datetime import datetime
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import torch


def plot_training_curves(history: Dict[str, list], save_dir: str):
    epochs = range(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="Train")
    plt.plot(epochs, history["test_loss"], label="Val")
    plt.title("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_acc"], label="Train")
    plt.plot(epochs, history["test_acc"], label="Val")
    plt.title("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_curves.png"))
    plt.close()

    with open(os.path.join(save_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=4)


def make_run_dir(cfg, hyp, base_dir="runs"):
    """
    Creates a deterministic run directory name from config and hyperparameters.
    If an identical configuration has already been run, reuse it.
    Returns:
        run_dir (str): Path to the run directory.
        exists (bool): True if the directory already existed, False if newly created.
    """
    run_name = (
        f"{hyp.starting_year}_{hyp.ending_year}/{cfg.name}/time_interval{hyp.time_interval}/goal_{hyp.goal_information}/"
        f"b{hyp.batch_size}_lr{hyp.learning_rate}_wr{hyp.weight_decay}_a{hyp.alpha}_b{hyp.beta}"
    )

    run_dir = os.path.join(base_dir, run_name)

    # Check if directory exists before creating
    exists = os.path.exists(run_dir)

    # Create if missing
    os.makedirs(run_dir, exist_ok=True)

    return run_dir, exists


def save_checkpoint(
    model,
    optimizer,
    epoch,
    best_acc,
    best_val_loss,
    history,
    run_dir,
    filename="checkpoint.pth",
):
    """Save model, optimizer, and training history."""
    checkpoint_path = os.path.join(run_dir, filename)
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_acc": best_acc,
            "best_val_loss": best_val_loss,
            "history": history,  # 👈 include training/test history
        },
        checkpoint_path,
    )
    print(f"💾 Checkpoint saved at {checkpoint_path}")


def load_checkpoint(model, optimizer, run_dir):
    """Load model, optimizer, and history if a checkpoint exists."""
    checkpoint_path = os.path.join(run_dir, "checkpoint.pth")
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])

        epoch = checkpoint.get("epoch", 0) + 1
        best_acc = checkpoint.get("best_acc", 0.0)
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        history = checkpoint.get(
            "history",
            {"train_loss": [], "test_loss": [], "train_acc": [], "test_acc": []},
        )

        print(f"✅ Resumed from checkpoint (epoch {epoch - 1})")
        return epoch, best_acc, best_val_loss, history

    # Default values if no checkpoint exists
    return (
        0,
        0.0,
        float("inf"),
        {"train_loss": [], "test_loss": [], "train_acc": [], "test_acc": []},
    )
