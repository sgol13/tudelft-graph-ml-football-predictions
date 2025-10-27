import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from experiment_configs import EXPERIMENTS

def train_one_epoch(model, dataloader, criterion, optimizer, device, forward_pass=None):
    model.train()
    total_loss = 0.0

    progress = tqdm(dataloader, desc="Training", leave=False)
    for batch in progress:
        batch = batch.to(device)
        optimizer.zero_grad()

        out, y = forward_pass(batch, model, device)

        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / len(dataloader)


@torch.no_grad()
def evaluate(model, dataloader, criterion, device, forward_pass=None):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    progress = tqdm(dataloader, desc="Evaluating", leave=False)
    for batch in progress:
        batch = batch.to(device)

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
    args = parser.parse_args()

    cfg = EXPERIMENTS[args.exp]
    print(f"Running experiment: {cfg.name}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dataset
    dataset = cfg.dataset
    print(f"Dataset size: {len(dataset)}")

    forward_pass = cfg.forward_pass_gat

    # Split into train/test
    train_size = int(cfg.train_split * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size], generator=torch.Generator().manual_seed(cfg.seed)
    )

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=True)

    # Model setup
    model = cfg.model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)

    epoch_progress = tqdm(range(cfg.num_epochs), desc="Epochs")

    best_acc = 0.0
    for _ in epoch_progress:
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, forward_pass)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device, forward_pass)
        best_acc = max(best_acc, test_acc)

        epoch_progress.set_postfix(
            {
                "train_loss": f"{train_loss:.4f}",
                "test_loss": f"{test_loss:.4f}",
                "test_acc": f"{test_acc:.2%}",
                "best_acc": f"{best_acc:.2%}",
            }
        )

    print("\nTraining finished!")
    print(f"Best test accuracy: {best_acc:.2%}")


if __name__ == "__main__":
    main()
