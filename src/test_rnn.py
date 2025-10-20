import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from dataloader_paired import SequentialSoccerDataset
from rnn import SimpleRNNModel

def graph_to_features(data):
    """
    Converts a graph-based match snapshot into a fixed-length feature vector.
    You can expand this later with more sophisticated stats.
    """
    home_nodes = data.home_x.size(0)
    away_nodes = data.away_x.size(0)
    home_passes = data.home_edge_index.size(1)
    away_passes = data.away_edge_index.size(1)
    current_goal_diff = data.current_home_goals - data.current_away_goals
    final_goal_diff = data.final_home_goals - data.final_away_goals

    # simple numeric features
    features = torch.tensor([
        home_nodes, away_nodes, 
        home_passes, away_passes, 
        current_goal_diff, final_goal_diff
    ], dtype=torch.float)

    return features


def group_by_match(dataset):
    """
    Groups dataset samples into sequences per match.
    Returns a list of sequences (each is a list of Data objects).
    """
    match_groups = {}
    for i in range(len(dataset)):
        data = dataset[i]
        key = f"{data.season}_{data.match_id}"
        if key not in match_groups:
            match_groups[key] = []
        match_groups[key].append(data)

    # Sort intervals in order of time (start_minute)
    for key in match_groups:
        match_groups[key].sort(key=lambda d: d.start_minute.item())

    return list(match_groups.values())

def create_dataloaders(dataset, batch_size=1):
    """
    Groups the dataset by match and returns train/test loaders.
    """
    all_sequences = group_by_match(dataset)
    train_size = int(0.8 * len(all_sequences))
    train_seqs = all_sequences[:train_size]
    test_seqs = all_sequences[train_size:]

    return train_seqs, test_seqs


def train_one_epoch(model, train_seqs, criterion, optimizer, device):
    model.train()
    total_loss = 0

    for seq in tqdm(train_seqs, desc="Training"):
        # Build the sequence tensor
        features = torch.stack([graph_to_features(d) for d in seq]).unsqueeze(0).to(device)  # [1, seq_len, input_size]
        target = torch.argmax(seq[-1].final_result).unsqueeze(0).to(device)  # class index

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
        features = torch.stack([graph_to_features(d) for d in seq]).unsqueeze(0).to(device)
        target = torch.argmax(seq[-1].final_result).unsqueeze(0).to(device)

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
    dataset = SequentialSoccerDataset(root='data', ending_year=2016)
    print(f"Loaded dataset with {len(dataset)} samples")

    # Group into sequences
    train_seqs, test_seqs = create_dataloaders(dataset)

    print(f"Train matches: {len(train_seqs)}, Test matches: {len(test_seqs)}")

    # Model setup
    input_size = 6           # from graph_to_features()
    hidden_size = 64
    num_layers = 1
    output_size = 3          # [HomeWin, Draw, AwayWin]

    model = SimpleRNNModel(input_size, hidden_size, num_layers, output_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Training loop
    epochs = 10
    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_seqs, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, test_seqs, criterion, device)

        print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss:.4f}, Test Loss: {val_loss:.4f}, Test Acc: {val_acc:.4f}")

    torch.save(model.state_dict(), "rnn_soccer_model.pt")


if __name__ == "__main__":
    main()