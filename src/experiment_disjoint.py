import json
import os
import pickle
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from gat import DisjointModel
from torch.utils.data import DataLoader, Subset
from torch_geometric.data import Data, Dataset, HeteroData
from tqdm import tqdm

from dataloader_paired import ProgressiveSoccerDataset, TemporalSequence


def create_empty_heterodata():
    """Create an empty HeteroData object for sequence padding."""
    empty = HeteroData()

    # Create minimal graph structure to avoid errors
    empty["home"].x = torch.zeros((1, 4), dtype=torch.float)  # 1 node with 4 features
    empty["away"].x = torch.zeros((1, 4), dtype=torch.float)  # 1 node with 4 features

    # Empty edges
    empty["home", "passes_to", "home"].edge_index = torch.zeros(
        (2, 0), dtype=torch.long
    )
    empty["away", "passes_to", "away"].edge_index = torch.zeros(
        (2, 0), dtype=torch.long
    )

    # Empty edge weights
    empty["home", "passes_to", "home"].edge_weight = torch.zeros(
        (0,), dtype=torch.float
    )
    empty["away", "passes_to", "away"].edge_weight = torch.zeros(
        (0,), dtype=torch.float
    )

    # Add dummy metadata to avoid attribute errors
    empty.y = torch.tensor([0, 0, 0], dtype=torch.float)  # Neutral label
    empty.start_minute = torch.tensor(-1, dtype=torch.long)  # Invalid time
    empty.end_minute = torch.tensor(-1, dtype=torch.long)
    empty.current_home_goals = torch.tensor(-1, dtype=torch.long)
    empty.current_away_goals = torch.tensor(-1, dtype=torch.long)

    return empty


def extract_global_features(data: HeteroData):

    # Extract metadata for this graph
    start_min = data.start_minute.float()
    end_min = data.end_minute.float()
    curr_home_goals = data.current_home_goals.float()
    curr_away_goals = data.current_away_goals.float()

    # Create feature vector for HOME team perspective
    home_features = torch.tensor(
        [
            start_min / 90.0,  # Normalized start time
            end_min / 90.0,  # Normalized end time
            curr_home_goals,  # Current home goals
            curr_away_goals,  # Current away goals
            curr_home_goals - curr_away_goals,  # Goal difference (home perspective)
            (curr_home_goals + curr_away_goals),  # Total goals so far
        ]
    )

    # Create feature vector for AWAY team perspective
    away_features = torch.tensor(
        [
            start_min / 90.0,  # Normalized start time
            end_min / 90.0,  # Normalized end time
            curr_away_goals,  # Current away goals (their perspective)
            curr_home_goals,  # Current home goals (opponent)
            curr_away_goals - curr_home_goals,  # Goal difference (away perspective)
            (curr_home_goals + curr_away_goals),  # Total goals so far
        ]
    )

    return home_features, away_features


def collate_fixed_windows(batch: List[TemporalSequence], window_size=5):
    """Collate function that returns [window_size, batch_size, ...] tensors."""

    labels_y, labels_home, labels_away = [], [], []

    x1_by_timestep = [[] for _ in range(window_size)]
    x2_by_timestep = [[] for _ in range(window_size)]
    edge1_by_timestep = [[] for _ in range(window_size)]
    edge2_by_timestep = [[] for _ in range(window_size)]
    edgew1_by_timestep = [[] for _ in range(window_size)]
    edgew2_by_timestep = [[] for _ in range(window_size)]
    batch1_by_timestep = [[] for _ in range(window_size)]
    batch2_by_timestep = [[] for _ in range(window_size)]
    norm1_by_timestep = [[] for _ in range(window_size)]
    norm2_by_timestep = [[] for _ in range(window_size)]

    global_idx = 0
    for sequence_idx, sequence in enumerate(batch):
        seq_len = sequence.sequence_length
        padding = [create_empty_heterodata() for _ in range(window_size - 1)]
        padded_seq = padding + sequence.hetero_data_sequence

        for i in range(seq_len):
            window = padded_seq[i : i + window_size]

            for timestep_idx, hetero in enumerate(window):

                home_x = hetero["home"].x
                away_x = hetero["away"].x
                home_e = hetero["home", "passes_to", "home"].edge_index
                away_e = hetero["away", "passes_to", "away"].edge_index
                home_w = hetero["home", "passes_to", "home"].edge_weight
                away_w = hetero["away", "passes_to", "away"].edge_weight

                home_feat, away_feat = extract_global_features(hetero)

                # ✅ VERIFICAR Y LIMPIAR NANs EN INPUTS
                if torch.isnan(home_x).any():
                    print(
                        f"⚠️ NaN in home_x at sequence {sequence_idx}, timestep {timestep_idx}"
                    )
                    home_x = torch.nan_to_num(home_x, nan=0.0)

                if torch.isnan(away_x).any():
                    print(
                        f"⚠️ NaN in away_x at sequence {sequence_idx}, timestep {timestep_idx}"
                    )
                    away_x = torch.nan_to_num(away_x, nan=0.0)

                if torch.isnan(home_feat).any():
                    print(
                        f"⚠️ NaN in home_feat at sequence {sequence_idx}, timestep {timestep_idx}"
                    )
                    home_feat = torch.nan_to_num(home_feat, nan=0.0)

                if torch.isnan(away_feat).any():
                    print(
                        f"⚠️ NaN in away_feat at sequence {sequence_idx}, timestep {timestep_idx}"
                    )
                    away_feat = torch.nan_to_num(away_feat, nan=0.0)

                home_batch = torch.full((home_x.size(0),), global_idx, dtype=torch.long)
                away_batch = torch.full((away_x.size(0),), global_idx, dtype=torch.long)

                x1_by_timestep[timestep_idx].append(home_x)
                x2_by_timestep[timestep_idx].append(away_x)
                edge1_by_timestep[timestep_idx].append(home_e)
                edge2_by_timestep[timestep_idx].append(away_e)
                edgew1_by_timestep[timestep_idx].append(home_w)
                edgew2_by_timestep[timestep_idx].append(away_w)
                batch1_by_timestep[timestep_idx].append(home_batch)
                batch2_by_timestep[timestep_idx].append(away_batch)
                norm1_by_timestep[timestep_idx].append(home_feat)
                norm2_by_timestep[timestep_idx].append(away_feat)

            labels_y.append(sequence.y)
            labels_home.append(sequence.final_home_goals)
            labels_away.append(sequence.final_away_goals)

            global_idx += 1

    batch_size = len(labels_y)

    x1_final = []
    x2_final = []
    edge1_final = []
    edge2_final = []
    edgew1_final = []
    edgew2_final = []
    batch1_final = []
    batch2_final = []
    xnorm1_final = []
    xnorm2_final = []

    for t in range(window_size):
        x1_cat = torch.cat(x1_by_timestep[t], dim=0)
        x2_cat = torch.cat(x2_by_timestep[t], dim=0)
        edge1_cat = torch.cat(edge1_by_timestep[t], dim=1)
        edge2_cat = torch.cat(edge2_by_timestep[t], dim=1)
        edgew1_cat = torch.cat(edgew1_by_timestep[t], dim=0)
        edgew2_cat = torch.cat(edgew2_by_timestep[t], dim=0)
        batch1_cat = torch.cat(batch1_by_timestep[t], dim=0)
        batch2_cat = torch.cat(batch2_by_timestep[t], dim=0)
        xnorm1_cat = torch.stack(norm1_by_timestep[t])
        xnorm2_cat = torch.stack(norm2_by_timestep[t])

        x1_final.append(x1_cat)
        x2_final.append(x2_cat)
        edge1_final.append(edge1_cat)
        edge2_final.append(edge2_cat)
        edgew1_final.append(edgew1_cat)
        edgew2_final.append(edgew2_cat)
        batch1_final.append(batch1_cat)
        batch2_final.append(batch2_cat)
        xnorm1_final.append(xnorm1_cat)
        xnorm2_final.append(xnorm2_cat)

    # print(f"DEBUG: Structure: [window_size={window_size}, batch_size={batch_size}, ...]")
    # print(f"DEBUG: x1_final[0] shape: {x1_final[0].shape}")
    # print(f"DEBUG: batch1_final[0] shape: {batch1_final[0].shape}, unique: {batch1_final[0].unique().shape[0]}")

    return {
        "x1": x1_final,  # [window_size, total_nodes_home, features]
        "x2": x2_final,  # [window_size, total_nodes_away, features]
        "edge_index1": edge1_final,  # [window_size, 2, total_edges_home]
        "edge_index2": edge2_final,  # [window_size, 2, total_edges_away]
        "edge_weight1": edgew1_final,  # [window_size, total_nodes_home]
        "edge_weight2": edgew2_final,  # [window_size, total_nodes_away]
        "batch1": batch1_final,  # [window_size, total_nodes_home]
        "batch2": batch2_final,  # [window_size, total_nodes_away]
        "x_norm2_1": torch.stack(xnorm1_final),  # [window_size, batch_size, 6]
        "x_norm2_2": torch.stack(xnorm2_final),  # [window_size, batch_size, 6]
        "labels_y": torch.stack(labels_y),
        "labels_home_goals": torch.stack(labels_home),
        "labels_away_goals": torch.stack(labels_away),
        "batch_size": batch_size,
        "window_size": window_size,
    }


def move_batch_to_device(batch, device):
    """Move all tensors (and nested lists of tensors) in a batch to the given device."""

    def move_to_device(obj):
        if isinstance(obj, torch.Tensor):
            return obj.to(device)
        elif isinstance(obj, list):
            return [move_to_device(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: move_to_device(value) for key, value in obj.items()}
        else:
            return obj

    return move_to_device(batch)


def save_checkpoint(
    model, optimizer, epoch, best_acc, best_val_loss, run_dir, filename="checkpoint.pth"
):
    """Save model and optimizer state."""
    checkpoint_path = os.path.join(run_dir, filename)
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_acc": best_acc,
            "best_val_loss": best_val_loss,
        },
        checkpoint_path,
    )
    print(f"💾 Checkpoint saved at {checkpoint_path}")


def load_checkpoint(model, optimizer, run_dir):
    """Load model/optimizer state if a checkpoint exists."""
    checkpoint_path = os.path.join(run_dir, "checkpoint.pth")
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        print(f"✅ Resumed from checkpoint (epoch {checkpoint['epoch']})")
        return (
            checkpoint["epoch"] + 1,
            checkpoint["best_acc"],
            checkpoint["best_val_loss"],
        )
    return 0, 0.0, float("inf")


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss, total_correct, total_samples = 0.0, 0, 0

    progress = tqdm(dataloader, desc="Training", leave=False)
    for batch_idx, batch in enumerate(progress):
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad()

        outputs = model(
            x1=batch["x1"],
            x2=batch["x2"],
            edge_index1=batch["edge_index1"],
            edge_index2=batch["edge_index2"],
            edge_weight1=batch["edge_weight1"],
            edge_weight2=batch["edge_weight2"],
            batch1=batch["batch1"],
            batch2=batch["batch2"],
            x_norm2_1=batch["x_norm2_1"],
            x_norm2_2=batch["x_norm2_2"],
            batch_size=batch["batch_size"],
            window_size=batch["window_size"],
        )

        # Safety checks
        if torch.isnan(
            torch.tensor([p.detach().float().mean() for p in model.parameters()])
        ).any():
            print(f"⚠️ NaN detected in model parameters at batch {batch_idx}")
            continue

        if model.goal_information:
            loss = criterion(outputs, batch)
            pred_logits = outputs["class_logits"]
        else:
            loss = criterion(outputs, batch["labels_y"])
            pred_logits = outputs

        if torch.isnan(loss):
            print(f"⚠️ NaN loss detected at batch {batch_idx}")
            continue

        # Backprop
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

        # Compute classification accuracy
        _, predicted = torch.max(pred_logits, 1)
        _, labels = torch.max(batch["labels_y"], 1)
        total_correct += (predicted == labels).sum().item()
        total_samples += labels.size(0)

        progress.set_postfix(loss=f"{loss.item():.4f}")

    acc = 100.0 * total_correct / total_samples if total_samples > 0 else 0.0
    avg_loss = total_loss / len(dataloader)
    return avg_loss, total_samples, total_correct, acc


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss, total_correct, total_samples = 0.0, 0, 0

    progress = tqdm(dataloader, desc="Evaluating", leave=False)
    for batch in progress:
        batch = move_batch_to_device(batch, device)

        outputs = model(
            x1=batch["x1"],
            x2=batch["x2"],
            edge_index1=batch["edge_index1"],
            edge_index2=batch["edge_index2"],
            edge_weight1=batch["edge_weight1"],
            edge_weight2=batch["edge_weight2"],
            batch1=batch["batch1"],
            batch2=batch["batch2"],
            x_norm2_1=batch["x_norm2_1"],
            x_norm2_2=batch["x_norm2_2"],
            batch_size=batch["batch_size"],
            window_size=batch["window_size"],
        )

        if model.goal_information:
            loss = criterion(outputs, batch)
            pred_logits = outputs["class_logits"]
        else:
            loss = criterion(outputs, batch["labels_y"])
            pred_logits = outputs

        total_loss += loss.item()

        _, predicted = torch.max(pred_logits, 1)
        _, labels = torch.max(batch["labels_y"], 1)
        total_correct += (predicted == labels).sum().item()
        total_samples += labels.size(0)

    acc = 100.0 * total_correct / total_samples if total_samples > 0 else 0.0
    avg_loss = total_loss / len(dataloader)
    return avg_loss, total_samples, total_correct, acc


# ============================================================
#                        LOSS WRAPPER
# ============================================================


def build_criterion(goal_information: bool, alpha: float = 1.0, beta: float = 0.5):
    """Return criterion callable with consistent signature."""
    if not goal_information:
        ce = nn.CrossEntropyLoss()
        return lambda outputs, batch: ce(outputs, batch["labels_y"])

    ce = nn.CrossEntropyLoss()
    poisson = nn.PoissonNLLLoss(log_input=False)
    eps = 1e-6

    def compute_loss(outputs: Dict[str, torch.Tensor], batch: Dict[str, Any]):
        loss_cls = ce(outputs["class_logits"], batch["labels_y"].argmax(dim=1))
        home_pred = outputs["home_goals_pred"].clamp_min(eps).squeeze()
        away_pred = outputs["away_goals_pred"].clamp_min(eps).squeeze()
        loss_home = poisson(home_pred, batch["labels_home_goals"].float().squeeze())
        loss_away = poisson(away_pred, batch["labels_away_goals"].float().squeeze())
        loss_goals = 0.5 * (loss_home + loss_away)
        return alpha * loss_cls + beta * loss_goals

    return compute_loss


# ============================================================
#                      PLOTTING UTILS
# ============================================================


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


import hashlib
import json
import os
from datetime import datetime


def make_run_dir(config, base_dir="runs"):
    """
    Creates a deterministic run directory name from config hyperparameters.
    If an identical configuration has already been run, reuse it.
    Otherwise, create a new folder.
    """
    # Stringify a subset of the config to make a unique but readable name
    short_cfg = {
        "model": config.get("model"),
        "ws": config.get("window_size"),
        "lr": config.get("learning_rate"),
        "goal": config.get("goal_information"),
        "alpha": config.get("alpha"),
        "beta": config.get("beta"),
    }

    # Hash to avoid excessively long names
    hash_suffix = hashlib.md5(
        json.dumps(short_cfg, sort_keys=True).encode()
    ).hexdigest()[:6]

    run_name = (
        f"{short_cfg['model']}_ws{short_cfg['ws']}_lr{short_cfg['lr']}_"
        f"goal{short_cfg['goal']}_a{short_cfg['alpha']}_b{short_cfg['beta']}_{hash_suffix}"
    )

    run_dir = os.path.join(base_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def main():
    # === CONFIG ==========================================================
    config = {
        "window_size": 3,
        "num_epochs": 40,
        "batch_size": 32,
        "learning_rate": 1e-4,
        "weight_decay": 1e-5,
        "patience": 5,
        "goal_information": True,
        "alpha": 1.0,
        "beta": 0.5,
        "model": "DisjointModel",
    }

    run_dir = make_run_dir(config)
    print(f"Run directory: {run_dir}")
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4)

    print("🚀 Starting training...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # === DATA ============================================================
    dataset = ProgressiveSoccerDataset(root="../data")
    print(f"Dataset size: {len(dataset)}")

    # Example split by season
    train_seasons = [
        "epl_2015",
        "epl_2016",
        "epl_2017",
        "epl_2018",
        "epl_2019",
        "epl_2020",
        "epl_2021",
        "epl_2022",
    ]
    val_seasons = ["epl_2023"]
    test_seasons = ["epl_2024"]

    train_idx, val_idx, test_idx = dataset.get_season_split(
        train_seasons, val_seasons, test_seasons
    )
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    test_dataset = Subset(dataset, test_idx)

    collate = lambda x: collate_fixed_windows(x, window_size=config["window_size"])

    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True, collate_fn=collate
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config["batch_size"], shuffle=False, collate_fn=collate
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config["batch_size"], shuffle=False, collate_fn=collate
    )

    # === MODEL ===========================================================
    model = DisjointModel(goal_information=config["goal_information"]).to(device)

    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.GRU):
            for name, param in m.named_parameters():
                if "weight" in name:
                    nn.init.orthogonal_(param)
                elif "bias" in name:
                    nn.init.constant_(param, 0)

    model.apply(init_weights)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = build_criterion(
        goal_information=config["goal_information"],
        alpha=config["alpha"],
        beta=config["beta"],
    )

    optimizer = optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )

    # === CHECKPOINT RESUME ==============================================
    start_epoch, best_acc, best_val_loss = load_checkpoint(model, optimizer, run_dir)

    history = {"train_loss": [], "test_loss": [], "train_acc": [], "test_acc": []}
    early_stopping_counter = 0

    # === TRAINING LOOP ===================================================
    for epoch in range(start_epoch, config["num_epochs"]):
        print(f"\nEpoch {epoch+1}/{config['num_epochs']}")

        train_loss, _, _, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, _, _, val_acc = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["test_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(val_acc)

        if not torch.isfinite(torch.tensor(train_loss)) or not torch.isfinite(
            torch.tensor(val_loss)
        ):
            print(f"⚠️ Epoch {epoch}: Invalid loss, skipping...")
            continue

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0
            torch.save(model.state_dict(), os.path.join(run_dir, "best_model.pth"))
            print(f"🏆 New best model saved (val_loss={val_loss:.4f})")
        else:
            early_stopping_counter += 1
            print(
                f"No improvement for {early_stopping_counter}/{config['patience']} epochs"
            )

        save_checkpoint(model, optimizer, epoch, best_acc, best_val_loss, run_dir)

        print(
            f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
            f"Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%"
        )

        if early_stopping_counter >= config["patience"]:
            print("🛑 Early stopping triggered.")
            break

    # === FINALIZE ========================================================
    plot_training_curves(history, run_dir)
    print("\n✅ Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Run directory: {run_dir}")


# ============================================================
#                        ENTRY POINT
# ============================================================
if __name__ == "__main__":
    main()
