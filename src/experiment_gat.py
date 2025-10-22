import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from dataloader_paired import SequentialSoccerDataset
from gat import SpatialModel


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0

    progress = tqdm(dataloader, desc="Training", leave=False)
    for batch in progress:
        batch = batch.to(device)
        optimizer.zero_grad()

        # Extract data from HeteroData structure
        x1 = batch["home"].x
        x2 = batch["away"].x
        edge_index1 = batch["home", "passes_to", "home"].edge_index
        edge_index2 = batch["away", "passes_to", "away"].edge_index

        # Get batch indices for each node type
        batch_idx1 = batch["home"].batch
        batch_idx2 = batch["away"].batch

        # Get batch size (number of graphs in this batch)
        batch_size = batch_idx1.max().item() + 1

        # Edge weights - compute mean per graph for global features
        edge_weight1 = batch["home", "passes_to", "home"].edge_weight
        edge_weight2 = batch["away", "passes_to", "away"].edge_weight

        # Compute mean edge weight per graph
        L = 16  # Feature dimension expected by model
        x_norm2_1 = torch.zeros(batch_size, L, device=device)
        x_norm2_2 = torch.zeros(batch_size, L, device=device)

        if edge_weight1.numel() > 0:
            for i in range(batch_size):
                # Find edges belonging to graph i
                mask = batch_idx1[edge_index1[0]] == i
                if mask.sum() > 0:
                    mean_weight = edge_weight1[mask].mean()
                    x_norm2_1[i] = mean_weight  # Broadcast to L dimensions

        if edge_weight2.numel() > 0:
            for i in range(batch_size):
                mask = batch_idx2[edge_index2[0]] == i
                if mask.sum() > 0:
                    mean_weight = edge_weight2[mask].mean()
                    x_norm2_2[i] = mean_weight

        # Optional edge attributes
        edge_col1 = None
        edge_col2 = None

        out = model(
            x1=x1,
            x2=x2,
            edge_index1=edge_index1,
            edge_index2=edge_index2,
            batch1=batch_idx1,
            batch2=batch_idx2,
            x_norm2_1=x_norm2_1,
            x_norm2_2=x_norm2_2,
            edge_col1=edge_col1,
            edge_col2=edge_col2,
        )

        # Get target labels - convert one-hot to class indices
        y = batch.y.argmax(dim=1)

        loss = criterion(out, y)
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

        # Same extraction as training
        x1 = batch["home"].x
        x2 = batch["away"].x
        edge_index1 = batch["home", "passes_to", "home"].edge_index
        edge_index2 = batch["away", "passes_to", "away"].edge_index

        batch_idx1 = batch["home"].batch
        batch_idx2 = batch["away"].batch

        batch_size = batch_idx1.max().item() + 1

        half_y = torch.stack(
            [
                torch.tensor(
                    [batch.current_home_goals[i], batch.current_away_goals[i]],
                    dtype=torch.float,
                    device=device,
                )
                for i in range(batch_size)
            ]
        )

        edge_weight1 = batch["home", "passes_to", "home"].edge_weight
        edge_weight2 = batch["away", "passes_to", "away"].edge_weight

        L = 16
        x_norm2_1 = torch.zeros(batch_size, L, device=device)
        x_norm2_2 = torch.zeros(batch_size, L, device=device)

        if edge_weight1.numel() > 0:
            for i in range(batch_size):
                mask = batch_idx1[edge_index1[0]] == i
                if mask.sum() > 0:
                    x_norm2_1[i] = edge_weight1[mask].mean()

        if edge_weight2.numel() > 0:
            for i in range(batch_size):
                mask = batch_idx2[edge_index2[0]] == i
                if mask.sum() > 0:
                    x_norm2_2[i] = edge_weight2[mask].mean()

        edge_col1 = None
        edge_col2 = None

        out = model(
            x1=x1,
            x2=x2,
            edge_index1=edge_index1,
            edge_index2=edge_index2,
            batch=batch_idx1,
            half_y=half_y,
            x_norm2_1=x_norm2_1,
            x_norm2_2=x_norm2_2,
            edge_col1=edge_col1,
            edge_col2=edge_col2,
        )

        y = batch.y.argmax(dim=1)

        loss = criterion(out, y)
        total_loss += loss.item()

        preds = out.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    accuracy = correct / total if total > 0 else 0
    return total_loss / len(dataloader), accuracy


def main():
    print("Starting training...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dataset
    dataset = SequentialSoccerDataset(root="../data")
    print(f"Dataset size: {len(dataset)}")

    # Split into train/test
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size]
    )
    print(f"Train size: {train_size}, Test size: {test_size}")

    # PyG's DataLoader works directly with HeteroData
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)  # type: ignore
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)  # type: ignore

    # Model setup
    input_size = 1  # Number of node features
    L = 16  # Additional features dimension
    model = SpatialModel(input_size=input_size, L=L).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 10
    epoch_progress = tqdm(range(num_epochs), desc="Epochs")

    best_acc = 0.0
    for _ in epoch_progress:
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        if test_acc > best_acc:
            best_acc = test_acc

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
