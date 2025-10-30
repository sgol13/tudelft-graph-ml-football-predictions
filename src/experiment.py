import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as GeometricDataLoader
from tqdm import tqdm

from dataloader_paired import TemporalSoccerDataset
from experiment_configs import EXPERIMENTS


def collate_temporal_sequences(data_list):
    """
    Collate function for TemporalSequence objects.

    Args:
        data_list: list of TemporalSequence objects (length = batch_size)

    Returns:
        A batch dictionary containing:
        - sequences: list of HeteroData sequences (one per match in batch)
        - labels: tensor of shape (batch_size,) with match outcomes
        - metadata: dict with additional info
    """
    # Extract sequences and labels
    sequences = [seq.hetero_data_sequence for seq in data_list]
    labels = torch.stack([seq.y for seq in data_list])

    # Optional: extract metadata
    metadata = {
        "final_home_goals": torch.stack([seq.final_home_goals for seq in data_list]),
        "final_away_goals": torch.stack([seq.final_away_goals for seq in data_list]),
        "sequence_lengths": torch.tensor([seq.sequence_length for seq in data_list]),
        "seasons": [seq.season for seq in data_list],
        "match_ids": [seq.match_id for seq in data_list],
        "indices": torch.tensor([seq.idx for seq in data_list]),
    }

    return {
        "sequences": sequences,
        "labels": labels,
        "metadata": metadata,
    }


def train_one_epoch(model, dataloader, criterion, optimizer, device, forward_pass):
    model.train()
    total_loss = 0.0

    progress = tqdm(dataloader, desc="Training", leave=False)
    for batch in progress:
        optimizer.zero_grad()

        out, y = forward_pass(batch, model, device)

        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / len(dataloader)


@torch.no_grad()
def evaluate(model, dataloader, criterion, device, forward_pass):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    progress = tqdm(dataloader, desc="Evaluating", leave=False)
    for batch in progress:

        out, y = forward_pass(batch, model, device)

        loss = criterion(out, y)
        total_loss += loss.item()

        preds = out.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    accuracy = correct / total if total > 0 else 0
    return total_loss / len(dataloader), accuracy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp",
        type=str,
        default="small",
        choices=EXPERIMENTS.keys(),
        help="Experiment configuration to run",
    )
    parser.add_argument(
        "--retrain-model",
        action="store_true",
        help="Whether you want to retrain an already trained model, if there is one. This will redo any training and overwrite the current saved model. Otherwise, the saved model will be used to evaluate.",
    )
    args = parser.parse_args()

    cfg = EXPERIMENTS[args.exp]
    print(f"Running experiment: {cfg.name}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dataset
    dataset = cfg.dataset_factory()
    print(f"Dataset size: {len(dataset)}")

    forward_pass = cfg.forward_pass

    # Split into train/test
    train_size = int(cfg.train_split * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(cfg.seed),
    )

    train_loader = GeometricDataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)  # type: ignore
    test_loader = GeometricDataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=True)  # type: ignore
    if type(dataset) is TemporalSoccerDataset:
        print("Using custom collate function")
        train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_temporal_sequences)  # type: ignore
        test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_temporal_sequences)  # type: ignore

    # Model setup
    model = cfg.model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    model_dict_path = f"model_backups/{cfg.name}.pt"
    should_load_model = (
        not args.retrain_model and Path.cwd().joinpath(model_dict_path).exists()
    )
    if should_load_model:
        model.load_state_dict(torch.load(model_dict_path))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)

    num_epochs = 1 if should_load_model else cfg.num_epochs
    epoch_progress = tqdm(range(num_epochs), desc="Epochs")

    best_acc = 0.0
    for _ in epoch_progress:
        train_loss = 0
        if not should_load_model:
            train_loss = train_one_epoch(
                model, train_loader, criterion, optimizer, device, forward_pass
            )
        test_loss, test_acc = evaluate(
            model, test_loader, criterion, device, forward_pass
        )
        best_acc = max(best_acc, test_acc)
        postfix_text = {
            "test_loss": f"{test_loss:.4f}",
            "test_acc": f"{test_acc:.2%}",
            "best_acc": f"{best_acc:.2%}",
        }
        if not should_load_model:
            postfix_text["train_loss"] = f"{train_loss:.4f}"

        epoch_progress.set_postfix()

    print("\nTraining finished!")
    print(f"Best test accuracy: {best_acc:.2%}")

    if args.retrain_model or not Path.cwd().joinpath(model_dict_path).exists():
        print("Saving model parameters...")
        torch.save(model.state_dict(), model_dict_path)


if __name__ == "__main__":
    main()
