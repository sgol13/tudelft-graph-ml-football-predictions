from dataclasses import dataclass
from typing import Callable

import torch

from dataloader_paired import SequentialSoccerDataset, SoccerDataset, CumulativeSoccerDataset
from models.gat import SpatialModel
from src.models.rnn import SimpleRNNModel


@dataclass
class ExperimentConfig:
    name: str
    dataset: SoccerDataset
    batch_size: int
    lr: float
    num_epochs: int
    model: torch.nn.Module
    forward_pass: Callable[[torch.Tensor, torch.nn.Module, torch.device], tuple[torch.Tensor, torch.Tensor]]
    train_split: float = 0.8
    seed = 42

def extract_global_features_alt(batch, batch_size, device):
    home_batch = batch["home"].batch
    away_batch = batch["away"].batch

    x_norm_home = []
    x_norm_away = []

    for i in range(batch_size):
        # Home
        home_mask = (home_batch == i)
        home_nodes = home_mask.sum()

        # select edges belonging to this home graph
        home_edge_index = batch["home", "passes_to", "home"].edge_index
        src_edge_home = home_edge_index[0]
        edge_mask_home = home_mask[src_edge_home]
        home_unique_passes = edge_mask_home.sum()
        home_total_passes = batch["home", "passes_to", "home"].edge_weight[edge_mask_home].sum()

        # Away
        away_mask = (away_batch == i)
        away_nodes = away_mask.sum()

        away_edge_index = batch["away", "passes_to", "away"].edge_index
        src_edge_away = away_edge_index[0]
        edge_mask_away = away_mask[src_edge_away]
        away_unique_passes = edge_mask_away.sum()
        away_total_passes = batch["away", "passes_to", "away"].edge_weight[edge_mask_away].sum()

        home_features = torch.tensor([home_nodes, home_unique_passes, home_total_passes], dtype=torch.float, device=device)
        away_features = torch.tensor([away_nodes, away_unique_passes, away_total_passes], dtype=torch.float, device=device)

        x_norm_home.append(home_features)
        x_norm_away.append(away_features)

    return torch.stack(x_norm_home), torch.stack(x_norm_away)

def forward_pass_rnn(batch, batch_size, device):
    pass

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

def forward_pass_gat(batch, model, device):
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

# Define multiple experiment setups here
EXPERIMENTS = {
    "small": ExperimentConfig(
        name="small",
        dataset=SequentialSoccerDataset(root="data", ending_year=2015),
        batch_size=16,
        lr=1e-3,
        num_epochs=1,
        model=SpatialModel(input_size=4, L=6),
        forward_pass=forward_pass_gat
    ),
    "large": ExperimentConfig(
        name="large",
        dataset=SequentialSoccerDataset(root="data"),
        batch_size=64,
        lr=5e-4,
        num_epochs=20,
        model=SpatialModel(input_size=4, L=6),
        forward_pass=forward_pass_gat
    ),
    "rnn": ExperimentConfig(
        name="rnn",
        dataset=CumulativeSoccerDataset(root="data"),
        batch_size=64,
        lr=5e-4,
        num_epochs=20,
        model=SimpleRNNModel(input_size=6, hidden_size=64, num_layers=1, output_size=3),
        forward_pass=forward_pass_rnn
    )
}
