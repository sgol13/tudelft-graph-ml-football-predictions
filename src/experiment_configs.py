from dataclasses import dataclass
from typing import Callable, Dict, Any

import torch

from dataloader_paired import (CumulativeSoccerDataset, SoccerDataset,
                               TemporalSoccerDataset)
from models.disjoint import DisjointModel
from models.gat import SpatialModel
from models.rnn import SimpleRNNModel
from models.varma import VARMABaseline
from criterion import build_criterion


@dataclass
class ExperimentConfig:
    name: str
    dataset_factory: Callable[[], SoccerDataset]
    model: torch.nn.Module
    forward_pass: Callable[
        [torch.Tensor, torch.nn.Module, torch.device], tuple[torch.Tensor, torch.Tensor]
    ]
    criterion: Callable[[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor],torch.Tensor
    ]
    train_split: float = 0.8
    seed = 42

@dataclass
class Hyperparameters:
    num_epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    patience: int 
    goal_information: bool
    alpha: float
    beta: float
    starting_year: int
    ending_year: int
    time_interval: int


def forward_pass_rnn(batch, model, device, percentage_of_match=0.8):
    """
    batch: dict with keys 'sequences', 'labels', 'metadata'
    sequences: list of HeteroData sequences (length = batch_size)
    """
    sequences = batch["sequences"]
    labels_y = batch["labels"].to(device).argmax(dim=1)
    labels_home_goals = batch["metadata"]["final_home_goals"].to(device)
    labels_away_goals = batch["metadata"]["final_away_goals"].to(device)

    all_features = []

    for match_sequence in sequences:  # Iterate over matches in the batch
        match_features = []
        stop = max(1, int(percentage_of_match * len(match_sequence)))

        for i in range(stop):
            data = match_sequence[i]

            # Extract features
            home_nodes = data["home"].x.size(0)
            away_nodes = data["away"].x.size(0)
            home_unique_passes = data["home", "passes_to", "home"].edge_index.size(1)
            away_unique_passes = data["away", "passes_to", "away"].edge_index.size(1)
            home_total_passes = (
                data["home", "passes_to", "home"].edge_weight.sum().item()
            )
            away_total_passes = (
                data["away", "passes_to", "away"].edge_weight.sum().item()
            )

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
                device=device,
            )
            match_features.append(features)

        # Stack features for this match: (seq_len, feature_dim)
        if len(match_features) > 0:
            match_features_tensor = torch.stack(match_features)
            all_features.append(match_features_tensor)

    if len(all_features) == 0:
        # Handle empty batch
        return torch.zeros(0, 3, device=device), torch.zeros(
            0, dtype=torch.long, device=device
        )

    # Pad sequences to same length for batching
    max_len = max(f.size(0) for f in all_features)
    padded_features = []

    for f in all_features:
        if f.size(0) < max_len:
            padding = torch.zeros(max_len - f.size(0), f.size(1), device=device)
            f = torch.cat([f, padding], dim=0)
        padded_features.append(f)

    # Stack into batch: (batch_size, seq_len, feature_dim)
    features_batch = torch.stack(padded_features)

    out = model(features_batch)

    return out, labels_y, labels_home_goals, labels_away_goals


def extract_global_feature_from_match(data, device):
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
    return home_features, away_features


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
    batch = batch.to(device)
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
    labels_y = batch.y.reshape(-1, 3).argmax(dim=1)
    labels_home_goals = batch.final_home_goals.reshape(-1,1) # Necessary?
    labels_away_goals = batch.final_away_goals.reshape(-1,1)

    return out, labels_y, labels_home_goals, labels_away_goals


def forward_pass_disjoint(batch, model, device, percentage_of_match=0.8):
    """
    batch: dict with keys 'sequences', 'labels', 'metadata'
    sequences: list of HeteroData sequences (length = batch_size)

    The DisjointModel expects:
    - x1, x2: lists of node features per timestep [window_size] each containing [total_nodes, 4]
    - edge_index1, edge_index2: lists of edge indices per timestep
    - edge_weight1, edge_weight2: lists of edge weights per timestep
    - batch1, batch2: lists of batch assignments per timestep
    - x_norm2_1, x_norm2_2: lists of global features per timestep [window_size] each containing [batch_size, 6]
    - batch_size, window_size: integers
    """
    sequences = batch["sequences"]
    labels_y = batch["labels"].to(device).argmax(dim=1)
    labels_home_goals = batch["metadata"]["final_home_goals"].to(device)
    labels_away_goals = batch["metadata"]["final_away_goals"].to(device)
    batch_size = len(sequences)

    # Determine window size (number of timeframes to use)
    window_size = max(1, int(percentage_of_match * len(sequences[0])))

    # Initialize lists to hold data for each timestep
    x1_list = []
    x2_list = []
    edge_index1_list = []
    edge_index2_list = []
    edge_weight1_list = []
    edge_weight2_list = []
    batch1_list = []
    batch2_list = []
    x_norm2_1_list = []
    x_norm2_2_list = []

    # Process each timestep across all matches
    for timestep in range(window_size):
        # Collect data from all matches at this timestep
        timestep_x1 = []
        timestep_x2 = []
        timestep_edge_index1 = []
        timestep_edge_index2 = []
        timestep_edge_weight1 = []
        timestep_edge_weight2 = []
        timestep_batch1 = []
        timestep_batch2 = []
        timestep_global_home = []
        timestep_global_away = []

        home_node_offset = 0
        away_node_offset = 0

        for match_idx, match_sequence in enumerate(sequences):
            # Get data for this match at this timestep
            if timestep < len(match_sequence):
                data = match_sequence[timestep]
            else:
                # If this match is shorter, use the last timeframe (padding)
                data = match_sequence[-1]

            # Extract node features
            home_nodes = data["home"].x.to(device)
            away_nodes = data["away"].x.to(device)

            # Extract edge information
            home_edge_index = data["home", "passes_to", "home"].edge_index.to(device)
            away_edge_index = data["away", "passes_to", "away"].edge_index.to(device)
            home_edge_weight = data["home", "passes_to", "home"].edge_weight.to(device)
            away_edge_weight = data["away", "passes_to", "away"].edge_weight.to(device)

            # Adjust edge indices for batching (add offset)
            home_edge_index_adjusted = home_edge_index + home_node_offset
            away_edge_index_adjusted = away_edge_index + away_node_offset

            # Create batch assignments
            home_batch = torch.full(
                (home_nodes.size(0),), match_idx, dtype=torch.long, device=device
            )
            away_batch = torch.full(
                (away_nodes.size(0),), match_idx, dtype=torch.long, device=device
            )

            # Extract global features
            home_feat, away_feat = extract_global_feature_from_match(data, device)

            # Append to timestep lists
            timestep_x1.append(home_nodes)
            timestep_x2.append(away_nodes)
            timestep_edge_index1.append(home_edge_index_adjusted)
            timestep_edge_index2.append(away_edge_index_adjusted)
            timestep_edge_weight1.append(home_edge_weight)
            timestep_edge_weight2.append(away_edge_weight)
            timestep_batch1.append(home_batch)
            timestep_batch2.append(away_batch)
            timestep_global_home.append(home_feat)
            timestep_global_away.append(away_feat)

            # Update offsets for next match
            home_node_offset += home_nodes.size(0)
            away_node_offset += away_nodes.size(0)

        # Concatenate all matches for this timestep
        x1_list.append(torch.cat(timestep_x1, dim=0))
        x2_list.append(torch.cat(timestep_x2, dim=0))
        edge_index1_list.append(torch.cat(timestep_edge_index1, dim=1))
        edge_index2_list.append(torch.cat(timestep_edge_index2, dim=1))
        edge_weight1_list.append(torch.cat(timestep_edge_weight1, dim=0))
        edge_weight2_list.append(torch.cat(timestep_edge_weight2, dim=0))
        batch1_list.append(torch.cat(timestep_batch1, dim=0))
        batch2_list.append(torch.cat(timestep_batch2, dim=0))
        x_norm2_1_list.append(torch.stack(timestep_global_home, dim=0))
        x_norm2_2_list.append(torch.stack(timestep_global_away, dim=0))

    # Forward pass through model
    out = model(
        x1=x1_list,
        x2=x2_list,
        edge_index1=edge_index1_list,
        edge_index2=edge_index2_list,
        edge_weight1=edge_weight1_list,
        edge_weight2=edge_weight2_list,
        batch1=batch1_list,
        batch2=batch2_list,
        x_norm2_1=x_norm2_1_list,
        x_norm2_2=x_norm2_2_list,
        batch_size=batch_size,
        window_size=window_size,
    )

    return out, labels_y, labels_home_goals, labels_away_goals


# Define hyperparameters
HYPERPARAMETERS = Hyperparameters(
    num_epochs=40,
    batch_size=32,
    learning_rate=5e-4,
    weight_decay=1e-5,
    patience=5, 
    goal_information=True,
    alpha=1.0,
    beta=0.5,
    starting_year=2015,
    ending_year=2024,
    time_interval=5
)

# Define multiple experiment setups here
EXPERIMENTS = {
    "small": ExperimentConfig(
        name="small",
        dataset_factory=lambda: CumulativeSoccerDataset(root="../data", ending_year=2015, time_interval=HYPERPARAMETERS.time_interval),
        model=SpatialModel(input_size=4, L=6, goal_information=HYPERPARAMETERS.goal_information),
        forward_pass=forward_pass_gat,
        criterion=build_criterion(goal_information=HYPERPARAMETERS.goal_information, alpha=HYPERPARAMETERS.alpha, beta=HYPERPARAMETERS.beta)
    ),
    "large": ExperimentConfig(
        name="large",
        dataset_factory=lambda: CumulativeSoccerDataset(root="../data", starting_year=HYPERPARAMETERS.starting_year, ending_year=HYPERPARAMETERS.ending_year, time_interval=HYPERPARAMETERS.time_interval),
        model=SpatialModel(input_size=4, L=6, goal_information=False),
        forward_pass=forward_pass_gat,
        criterion=build_criterion(goal_information=False)
    ),
    "rnn": ExperimentConfig(
        name="rnn",
        dataset_factory=lambda: TemporalSoccerDataset(root="../data", starting_year=HYPERPARAMETERS.starting_year, ending_year=HYPERPARAMETERS.ending_year, time_interval=HYPERPARAMETERS.time_interval),
        model=SimpleRNNModel(input_size=6, hidden_size=64, num_layers=1, output_size=3, goal_information=HYPERPARAMETERS.goal_information),
        forward_pass=forward_pass_rnn,
        criterion=build_criterion(goal_information=HYPERPARAMETERS.goal_information, alpha=HYPERPARAMETERS.alpha, beta=HYPERPARAMETERS.beta)
    ),
    "varma": ExperimentConfig(
        name="varma",
        dataset_factory=lambda: TemporalSoccerDataset(root="../data", starting_year=HYPERPARAMETERS.starting_year, ending_year=HYPERPARAMETERS.ending_year, time_interval=HYPERPARAMETERS.time_interval),
        model=VARMABaseline(input_size=6, hidden_size=64, p=2, q=1, num_classes=3, goal_information=HYPERPARAMETERS.goal_information),
        forward_pass=forward_pass_rnn,
        criterion=build_criterion(goal_information=HYPERPARAMETERS.goal_information, alpha=HYPERPARAMETERS.alpha, beta=HYPERPARAMETERS.beta)
    ),
    "disjoint": ExperimentConfig(
        name="disjoint",
        dataset_factory=lambda: TemporalSoccerDataset(root="../data", starting_year=HYPERPARAMETERS.starting_year, ending_year=HYPERPARAMETERS.ending_year, time_interval=HYPERPARAMETERS.time_interval),
        model=DisjointModel(goal_information=HYPERPARAMETERS.goal_information),
        forward_pass=forward_pass_disjoint,
        criterion=build_criterion(goal_information=HYPERPARAMETERS.goal_information, alpha=HYPERPARAMETERS.alpha, beta=HYPERPARAMETERS.beta)
    ),
}
