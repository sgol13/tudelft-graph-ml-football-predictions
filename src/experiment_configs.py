from dataclasses import dataclass
from typing import Callable, Dict

import torch

from criterion import build_criterion
from dataloader_paired import (CumulativeSoccerDataset, SoccerDataset,
                               TemporalSoccerDataset)
from models.disjoint import DisjointModel
from models.gat import SpatialModel
from models.rnn import SimpleRNNModel
from models.varma import VARMABaseline
from src.dataloader_paired import TemporalSequence


@dataclass
class ExperimentConfig:
    name: str
    dataset_factory: Callable[[], SoccerDataset]
    model: torch.nn.Module
    forward_pass: Callable[
        [torch.Tensor, torch.nn.Module, torch.device],
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ]
    criterion: Callable[
        [Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor],
        torch.Tensor,
    ]
    train_split: float = 0.8
    seed = 42


@dataclass
class Hyperparameters:
    num_epochs: int
    learning_rate: float
    weight_decay: float
    patience: int
    goal_information: bool
    alpha: float
    beta: float
    starting_year: int
    ending_year: int
    time_interval: int


def forward_pass_rnn(entry: TemporalSequence, model, device, percentage_of_match=0.8):
    """
    batch: dict with keys 'sequence', 'labels', 'metadata'
    sequence: list of HeteroData sequence (length = batch_size)
    """
    sequence = entry.hetero_data_sequence
    labels_y = entry.y.to(device).argmax(dim=0).unsqueeze(0)
    labels_home_goals = entry.final_home_goals.to(device).unsqueeze(0)
    labels_away_goals = entry.final_away_goals.to(device).unsqueeze(0)

    match_features = []
    stop = max(1, int(percentage_of_match * len(sequence)))

    for i in range(stop):
        data = sequence[i]

        # Extract features
        home_nodes = data["home"].x.size(0)
        away_nodes = data["away"].x.size(0)
        home_unique_passes = data["home", "passes_to", "home"].edge_index.size(1)
        away_unique_passes = data["away", "passes_to", "away"].edge_index.size(1)
        home_total_passes = data["home", "passes_to", "home"].edge_weight.sum().item()
        away_total_passes = data["away", "passes_to", "away"].edge_weight.sum().item()

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

    # Stack features for this match: (1, seq_len, feature_dim)
    match_features_tensor = torch.stack(match_features).unsqueeze(0)

    # out: (1, 3)
    out = model(match_features_tensor)

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


def forward_pass_gat(entry, model, device):
    entry = entry.to(device)
    # Extract data from HeteroData structure
    x1 = entry["home"].x
    x2 = entry["away"].x
    edge_index1 = entry["home", "passes_to", "home"].edge_index
    edge_index2 = entry["away", "passes_to", "away"].edge_index

    # Edge weights
    edge_weight1 = entry["home", "passes_to", "home"].edge_weight
    edge_weight2 = entry["away", "passes_to", "away"].edge_weight

    normalized_edge_weight1 = (
        edge_weight1 / (edge_weight1.max() + 1e-8)
        if edge_weight1.numel() > 0
        else torch.zeros(edge_weight1.size(), device=device)
    )
    normalized_edge_weight2 = (
        edge_weight2 / (edge_weight2.max() + 1e-8)
        if edge_weight2.numel() > 0
        else torch.zeros(edge_weight2.size(), device=device)
    )

    # Extract global features from metadata
    x_norm2_1, x_norm2_2 = extract_global_feature_from_match(entry, device)

    # out shape: (3,)
    out = model(
        x1=x1,
        x2=x2,
        edge_index1=edge_index1,
        edge_index2=edge_index2,
        x_norm2_1=x_norm2_1,
        x_norm2_2=x_norm2_2,
        edge_col1=normalized_edge_weight1,
        edge_col2=normalized_edge_weight2,
    )

    # y shape: (1,)
    labels_y = entry.y.reshape(-1, 3).argmax(dim=1)
    labels_home_goals = entry.final_home_goals.reshape(-1, 1)  # Necessary?
    labels_away_goals = entry.final_away_goals.reshape(-1, 1)

    return out, labels_y, labels_home_goals, labels_away_goals


def forward_pass_disjoint(
    entry: TemporalSequence, model: DisjointModel, device, percentage_of_match=0.8
):
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
    sequence = entry.hetero_data_sequence
    labels_y = entry.y.to(device).argmax(dim=0).unsqueeze(0)
    labels_home_goals = entry.final_home_goals.to(device).unsqueeze(0)
    labels_away_goals = entry.final_away_goals.to(device).unsqueeze(0)

    # Determine window size (number of timeframes to use)
    window_size = max(1, int(percentage_of_match * len(sequence)))

    x1_list, x2_list = [], []
    edge_index1_list, edge_index2_list = [], []
    edge_weight1_list, edge_weight2_list = [], []
    x_norm2_1_list, x_norm2_2_list = [], []

    # Process each timestep
    for t in range(window_size):
        timeframe = sequence[t].to(device)
        # Extract data from HeteroData structure
        x1_list.append(timeframe["home"].x)
        x2_list.append(timeframe["away"].x)
        edge_index1_list.append(timeframe["home", "passes_to", "home"].edge_index)
        edge_index2_list.append(timeframe["away", "passes_to", "away"].edge_index)

        # Edge weights
        edge_weight1 = timeframe["home", "passes_to", "home"].edge_weight
        edge_weight2 = timeframe["away", "passes_to", "away"].edge_weight

        normalized_edge_weight1 = (
            edge_weight1 / (edge_weight1.max() + 1e-8)
            if edge_weight1.numel() > 0
            else torch.zeros(edge_weight1.size(), device=device)
        )
        normalized_edge_weight2 = (
            edge_weight2 / (edge_weight2.max() + 1e-8)
            if edge_weight2.numel() > 0
            else torch.zeros(edge_weight2.size(), device=device)
        )

        edge_weight1_list.append(normalized_edge_weight1)
        edge_weight2_list.append(normalized_edge_weight2)

        # Extract global features from metadata
        x_norm2_1, x_norm2_2 = extract_global_feature_from_match(timeframe, device)
        x_norm2_1_list.append(x_norm2_1)
        x_norm2_2_list.append(x_norm2_2)

    out = model(
        x1=x1_list,
        x2=x2_list,
        edge_index1=edge_index1_list,
        edge_index2=edge_index2_list,
        edge_weight1=edge_weight1_list,
        edge_weight2=edge_weight2_list,
        x_norm2_1=x_norm2_1_list,
        x_norm2_2=x_norm2_2_list,
        window_size=window_size,
    )

    return out, labels_y, labels_home_goals, labels_away_goals


# Define hyperparameters
HYPERPARAMETERS = Hyperparameters(
    num_epochs=20,
    learning_rate=5e-4,
    weight_decay=1e-5,
    patience=10,
    goal_information=True,
    alpha=1.0,
    beta=0.5,
    starting_year=2020,
    ending_year=2024,
    time_interval=5,
)

# Define multiple experiment setups here
EXPERIMENTS = {
    "small": ExperimentConfig(
        name="small",
        dataset_factory=lambda: CumulativeSoccerDataset(
            root="data",
            ending_year=2015,
            time_interval=HYPERPARAMETERS.time_interval,
        ),
        model=SpatialModel(
            input_size=4, L=6, goal_information=HYPERPARAMETERS.goal_information
        ),
        forward_pass=forward_pass_gat,
        criterion=build_criterion(
            goal_information=HYPERPARAMETERS.goal_information,
            alpha=HYPERPARAMETERS.alpha,
            beta=HYPERPARAMETERS.beta,
        ),
    ),
    "large": ExperimentConfig(
        name="large",
        dataset_factory=lambda: CumulativeSoccerDataset(
            root="data",
            starting_year=HYPERPARAMETERS.starting_year,
            ending_year=HYPERPARAMETERS.ending_year,
            time_interval=HYPERPARAMETERS.time_interval,
        ),
        model=SpatialModel(input_size=4, L=6, goal_information=False),
        forward_pass=forward_pass_gat,
        criterion=build_criterion(goal_information=False),
    ),
    "rnn": ExperimentConfig(
        name="rnn",
        dataset_factory=lambda: TemporalSoccerDataset(
            root="data",
            starting_year=HYPERPARAMETERS.starting_year,
            ending_year=HYPERPARAMETERS.ending_year,
            time_interval=HYPERPARAMETERS.time_interval,
        ),
        model=SimpleRNNModel(
            input_size=6,
            hidden_size=64,
            num_layers=1,
            output_size=3,
            goal_information=HYPERPARAMETERS.goal_information,
        ),
        forward_pass=forward_pass_rnn,
        criterion=build_criterion(
            goal_information=HYPERPARAMETERS.goal_information,
            alpha=HYPERPARAMETERS.alpha,
            beta=HYPERPARAMETERS.beta,
        ),
    ),
    "varma": ExperimentConfig(
        name="varma",
        dataset_factory=lambda: TemporalSoccerDataset(
            root="data",
            starting_year=HYPERPARAMETERS.starting_year,
            ending_year=HYPERPARAMETERS.ending_year,
            time_interval=HYPERPARAMETERS.time_interval,
        ),
        model=VARMABaseline(
            input_size=6,
            hidden_size=64,
            p=2,
            q=1,
            num_classes=3,
            goal_information=HYPERPARAMETERS.goal_information,
        ),
        forward_pass=forward_pass_rnn,
        criterion=build_criterion(
            goal_information=HYPERPARAMETERS.goal_information,
            alpha=HYPERPARAMETERS.alpha,
            beta=HYPERPARAMETERS.beta,
        ),
    ),
    "disjoint": ExperimentConfig(
        name="disjoint",
        dataset_factory=lambda: TemporalSoccerDataset(
            root="data",
            starting_year=HYPERPARAMETERS.starting_year,
            ending_year=HYPERPARAMETERS.ending_year,
            time_interval=HYPERPARAMETERS.time_interval,
        ),
        model=DisjointModel(goal_information=HYPERPARAMETERS.goal_information),
        forward_pass=forward_pass_disjoint,
        criterion=build_criterion(
            goal_information=HYPERPARAMETERS.goal_information,
            alpha=HYPERPARAMETERS.alpha,
            beta=HYPERPARAMETERS.beta,
        ),
    ),
}
