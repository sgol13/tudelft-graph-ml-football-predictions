import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from dataloader_paired import SequentialSoccerDataset
from src.models.gat import SpatialModel


def extract_global_features(batch, batch_size, device):
    """
    Extract global features from batch metadata for each graph.
    Features include: time info and current score.
    """
    x_norm_home = []
    x_norm_away = []

    for i in range(batch_size):
        # Extract metadata for this graph
        start_min = batch.start_minute[i].float()
        end_min = batch.end_minute[i].float()
        curr_home_goals = batch.current_home_goals[i].float()
        curr_away_goals = batch.current_away_goals[i].float()

        # Create feature vector for HOME team perspective
        home_features = torch.tensor(
            [
                start_min / 90.0,  # Normalized start time
                end_min / 90.0,  # Normalized end time
                curr_home_goals,  # Current home goals
                curr_away_goals,  # Current away goals
                curr_home_goals - curr_away_goals,  # Goal difference (home perspective)
                (curr_home_goals + curr_away_goals),  # Total goals so far
            ],
            device=device,
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
            ],
            device=device,
        )

        x_norm_home.append(home_features)
        x_norm_away.append(away_features)

    return torch.stack(x_norm_home), torch.stack(x_norm_away)


def forward_pass(batch, model, device):
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

    # Edge weights
    edge_weight1 = batch["home", "passes_to", "home"].edge_weight
    edge_weight2 = batch["away", "passes_to", "away"].edge_weight

    normalized_edge_weight1 = edge_weight1 / (edge_weight1.max() + 1e-8)
    normalized_edge_weight2 = edge_weight2 / (edge_weight2.max() + 1e-8)

    # Extract global features from metadata
    x_norm2_1, x_norm2_2 = extract_global_features(batch, batch_size, device)

    # out shape: (batch_size, 3)
    out = model(
        x1=x1,
        x2=x2,
        edge_index1=edge_index1,
        edge_index2=edge_index2,
        batch1=batch_idx1,
        batch2=batch_idx2,
        x_norm2_1=x_norm2_1,
        x_norm2_2=x_norm2_2,
        edge_col1=normalized_edge_weight1,
        edge_col2=normalized_edge_weight2,
    )

    # y shape: (batch_size,)
    y = batch.y.reshape(-1, 3).argmax(dim=1)
    return out, y


def train_one_epoch(model, dataloader, criterion, optimizer, device):
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
def evaluate(model, dataloader, criterion, device):
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
    input_size = 4  # Number of node features
    L = 6  # Additional features dimension
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
