from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

from dataloader_paired import SequentialSoccerDataset
from src.models.rnn import SimpleRNNModel


def graph_to_features(data):
    """
    Converts a graph-based match snapshot into a fixed-length feature vector.
    You can expand this later with more sophisticated stats.
    """
    home_nodes = data["home"].x.size(0)
    away_nodes = data["away"].x.size(0)
    home_unique_passes = data["home", "passes_to", "home"].edge_index.size(1)
    away_unique_passes = data["away", "passes_to", "away"].edge_index.size(1)
    home_total_passes = data["home", "passes_to", "home"].edge_weight.sum()
    away_total_passes = data["away", "passes_to", "away"].edge_weight.sum()

    # simple numeric features
    features = torch.tensor(
        [
            home_nodes,
            away_nodes,
            home_unique_passes,
            away_unique_passes,
            home_total_passes,
            away_total_passes,
        ],
        dtype=torch.float,
    )

    return features


def load_all_data(dataset):
    data_list = []
    for i in range(len(dataset)):
        file = Path(dataset.processed_dir) / f"data_{i}.pt"
        data_list.append(torch.load(file, weights_only=False))
    return data_list


# Now group by match in memory
def group_by_match(data_list):
    match_groups = {}
    for data in data_list:
        key = f"{data.season}_{data.match_id}"
        if key not in match_groups:
            match_groups[key] = []
        match_groups[key].append(data)
    return match_groups


def split_sequence(seq, train_ratio=0.8):
    """
    Split a single match sequence into train/test based on snapshots.
    Returns train_seq, test_seq
    """
    n = len(seq)
    split_idx = int(n * train_ratio)
    return seq[:split_idx], seq[split_idx:]


def create_dataloaders(dataset):
    """
    Group dataset by match and split each match sequence 80/20.
    Returns train_seqs and test_seqs (lists of sequences).
    """
    data_list = load_all_data(dataset)

    # Group by match
    match_groups = group_by_match(data_list)

    train_seqs = []
    test_seqs = []

    for seq in match_groups.values():
        train_part, test_part = split_sequence(seq)
        if len(train_part) > 0:
            train_seqs.append(train_part)
        if len(test_part) > 0:
            test_seqs.append(test_part)

    return train_seqs, test_seqs


def train_one_epoch(model, train_seqs, criterion, optimizer, device):
    model.train()
    total_loss = 0

    for seq in tqdm(train_seqs, desc="Training"):
        # Build the sequence tensor
        features = (
            torch.stack([graph_to_features(d) for d in seq]).unsqueeze(0).to(device)
        )  # [1, seq_len, input_size]
        target = torch.argmax(seq[-1]["y"]).unsqueeze(0).to(device)  # class index

        optimizer.zero_grad()
        output = model(features)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_seqs)


@torch.no_grad()
def evaluate(model, test_seqs, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    for seq in tqdm(test_seqs, desc="Evaluating"):
        features = (
            torch.stack([graph_to_features(d) for d in seq]).unsqueeze(0).to(device)
        )
        target = torch.argmax(seq[-1]["y"]).unsqueeze(0).to(device)

        output = model(features)
        loss = criterion(output, target)
        total_loss += loss.item()

        preds = torch.argmax(output, dim=1)
        correct += (preds == target).sum().item()
        total += target.size(0)

    acc = correct / total if total > 0 else 0
    return total_loss / len(test_seqs), acc


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset
    dataset = SequentialSoccerDataset(root="data", ending_year=2015)
    print(f"Loaded dataset with {len(dataset)} samples")

    # Group into sequences
    train_seqs, test_seqs = create_dataloaders(dataset)

    print(f"Train matches: {len(train_seqs)}, Test matches: {len(test_seqs)}")

    # Model setup
    input_size = 6  # from graph_to_features()
    hidden_size = 64
    num_layers = 1
    output_size = 3  # [HomeWin, Draw, AwayWin]

    model = SimpleRNNModel(input_size, hidden_size, num_layers, output_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Training loop
    epochs = 10
    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_seqs, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, test_seqs, criterion, device)

        print(
            f"Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss:.4f}, Test Loss: {val_loss:.4f}, Test Acc: {val_acc:.4f}"
        )

    torch.save(model.state_dict(), "rnn_soccer_model.pt")


if __name__ == "__main__":
    main()
