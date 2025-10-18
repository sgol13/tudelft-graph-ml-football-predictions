import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from dataloader_paired import SequentialSoccerDataset
from gat import GAT


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0

    progress = tqdm(dataloader, desc="Training", leave=False)
    for batch in progress:
        batch = batch.to(device)
        optimizer.zero_grad()

        # Forward pass (adjust depending on your GAT forward signature)
        out = model(batch.x, batch.edge_index, batch.batch)
        loss = criterion(out, batch.y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / len(dataloader)


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    progress = tqdm(dataloader, desc="Evaluating", leave=False)
    for batch in progress:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.batch)
        loss = criterion(out, batch.y)
        total_loss += loss.item()

        preds = out.argmax(dim=1)
        correct += (preds == batch.y).sum().item()
        total += batch.y.size(0)

    accuracy = correct / total if total > 0 else 0
    return total_loss / len(dataloader), accuracy


def main():
    print("Starting training...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dataset
    dataset = SequentialSoccerDataset(root="../data")

    # Split into train/test
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=1)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=1)

    # Model setup
    model = GAT().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 10
    epoch_progress = tqdm(range(num_epochs), desc="Epochs")

    for _ in epoch_progress:
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        epoch_progress.set_postfix(
            {
                "train_loss": f"{train_loss:.4f}",
                "test_loss": f"{test_loss:.4f}",
                "test_acc": f"{test_acc:.2%}",
            }
        )

    print("Training finished!")


if __name__ == "__main__":
    main()
