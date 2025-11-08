from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Dict

import torch
from torch_geometric.data import HeteroData, Batch, Data

from criterion import build_criterion
from dataloader_paired import (CumulativeSoccerDataset, SoccerDataset,
                               TemporalSequence, TemporalSoccerDataset, TemporalAllPlayersSoccerDataset)
from dataloader_paired import (
    CumulativeSoccerDataset,
    SoccerDataset,
    TemporalSequence,
    TemporalSoccerDataset,
)
from models.disjoint import DisjointModel
from models.gat import SpatialModel
from models.rnn import SimpleRNNModel
from models.varma import VARMABaseline
from models.graph_rnn import GraphRNNModel
from models.no_goals import NoGoalsModel
from models.product_graphs import ProductGraphsModel


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
    only_cpu: bool = False
    trainable: bool = True


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


def forward_pass_rnn(entry: TemporalSequence, model, device, percentage_of_match=1):
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

    # Set the y data to match with the stop
    labels_y = labels_y.repeat(1, stop).reshape(-1)
    labels_home_goals = labels_home_goals.repeat(1, stop).reshape(-1)
    labels_away_goals = labels_away_goals.repeat(1, stop).reshape(-1)

    for i in range(stop):
        data = sequence[i]

        home_features, away_features = extract_global_feature_from_match(data, device)
        match_features.append(torch.cat([home_features, away_features], dim=0))

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
    entry: TemporalSequence, model: DisjointModel, device, percentage_of_match=1.0
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

    # Set the y data to match with the window_size
    labels_y = labels_y.repeat(1, window_size).reshape(-1)
    labels_home_goals = labels_home_goals.repeat(1, window_size).reshape(-1)
    labels_away_goals = labels_away_goals.repeat(1, window_size).reshape(-1)

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


def normalize_edge_weights(edge_weights: torch.Tensor) -> torch.Tensor:
    return (
        edge_weights / (edge_weights.max() + 1e-8)
        if edge_weights.numel() > 0
        else torch.zeros(edge_weights.size(), device=edge_weights.device)
    )


def forward_pass_graph_rnn(
    entry: TemporalSequence, model: GraphRNNModel, device, percentage_of_match=1.0
):
    sequence = entry.hetero_data_sequence
    labels_y = entry.y.to(device).argmax(dim=0).unsqueeze(0)
    labels_home_goals = entry.final_home_goals.to(device).unsqueeze(0)
    labels_away_goals = entry.final_away_goals.to(device).unsqueeze(0)

    window_size = max(1, int(percentage_of_match * len(sequence)))

    labels_y = labels_y.repeat(1, window_size).reshape(-1)
    labels_home_goals = labels_home_goals.repeat(1, window_size).reshape(-1)
    labels_away_goals = labels_away_goals.repeat(1, window_size).reshape(-1)

    home_graphs_list: list[Data] = []
    away_graphs_list: list[Data] = []
    home_features_list: list[torch.Tensor] = []
    away_features_list: list[torch.Tensor] = []

    for t in range(window_size):
        timeframe = sequence[t].to(device)

        # extract graphs
        data_home = Data(
            x=timeframe["home"].x,
            edge_index=timeframe["home", "passes_to", "home"].edge_index,
            edge_attr=normalize_edge_weights(timeframe["home", "passes_to", "home"].edge_weight),
        ).to(device)
        home_graphs_list.append(data_home)

        data_away = Data(
            x=timeframe["away"].x,
            edge_index=timeframe["away", "passes_to", "away"].edge_index,
            edge_attr=normalize_edge_weights(timeframe["away", "passes_to", "away"].edge_weight),
        ).to(device)
        away_graphs_list.append(data_away)

        # extract global team features
        home_features, away_features = extract_global_feature_from_match(timeframe, device)
        home_features_list.append(home_features)
        away_features_list.append(away_features)

    out = model(
        home_graphs_list,
        away_graphs_list,
        home_features_list,
        away_features_list,
        window_size=window_size,
    )

    return out, labels_y, labels_home_goals, labels_away_goals


def forward_pass_no_goals_baseline(
    entry: TemporalSequence, model: NoGoalsModel, device, percentage_of_match=1.0
):
    sequence = entry.hetero_data_sequence
    labels_y = entry.y.to(device).argmax(dim=0).unsqueeze(0)
    labels_home_goals = entry.final_home_goals.to(device).unsqueeze(0)
    labels_away_goals = entry.final_away_goals.to(device).unsqueeze(0)

    window_size = max(1, int(percentage_of_match * len(sequence)))

    labels_y = labels_y.repeat(1, window_size).reshape(-1)
    labels_home_goals = labels_home_goals.repeat(1, window_size).reshape(-1)
    labels_away_goals = labels_away_goals.repeat(1, window_size).reshape(-1)

    out = model(entry, window_size=window_size)

    return out, labels_y, labels_home_goals, labels_away_goals


def build_product_graph(subseq: list[Data]) -> Data:
    # combines into one graph, not a real batch
    temporal_graph = Batch.from_data_list(subseq)

    spatial_edges = temporal_graph.edge_index
    spatial_edge_attrs = temporal_graph.edge_attr
    num_nodes = subseq[0].num_nodes

    # extract inter-timestamp edges
    cum_nodes_offset = 0
    inter_timestamp_edges = []
    inter_timestamp_attrs = []

    for t in range(len(subseq) - 1):
        assert num_nodes == subseq[t + 1].num_nodes

        src_nodes = torch.arange(
            cum_nodes_offset,
            cum_nodes_offset + num_nodes,
            dtype=torch.long
        )
        dst_nodes = torch.arange(
            cum_nodes_offset + num_nodes,
            cum_nodes_offset + 2 * num_nodes,
            dtype=torch.long
        )

        inter_timestamp_edges.append(torch.stack([src_nodes, dst_nodes]))

        # Set inter-timestamp edge weights as the sum of spatial edge weights at time t divided by the number of nodes
        sum_of_weights = subseq[t].edge_attr.sum()
        new_attr_value = sum_of_weights / num_nodes if num_nodes > 0 else 0.0

        new_temporal_attrs = torch.full(
            size=(num_nodes,),
            fill_value=new_attr_value.item(),
            dtype=spatial_edge_attrs.dtype,
            device=spatial_edge_attrs.device
        )
        inter_timestamp_attrs.append(new_temporal_attrs)

        cum_nodes_offset += num_nodes

    # combine spatial and inter-timestamp edges
    if inter_timestamp_edges:
        all_temporal_edges = torch.cat(inter_timestamp_edges, dim=1)
        all_temporal_attrs = torch.cat(inter_timestamp_attrs, dim=0)

        temporal_graph.edge_index = torch.cat(
            [spatial_edges, all_temporal_edges],
            dim=1
        )
        temporal_graph.edge_attr = torch.cat(
            [spatial_edge_attrs, all_temporal_attrs],
            dim=0
        )

    assert temporal_graph.edge_index.shape[1] == temporal_graph.edge_attr.shape[0]

    return temporal_graph


def forward_pass_product_graphs(
    entry: TemporalSequence, model: ProductGraphsModel, device, percentage_of_match=1.0, product_length: int = 5
):
    sequence = entry.hetero_data_sequence
    labels_y = entry.y.to(device).argmax(dim=0).unsqueeze(0)
    labels_home_goals = entry.final_home_goals.to(device).unsqueeze(0)
    labels_away_goals = entry.final_away_goals.to(device).unsqueeze(0)

    window_size = max(1, int(percentage_of_match * len(sequence)))

    labels_y = labels_y.repeat(1, window_size).reshape(-1)
    labels_home_goals = labels_home_goals.repeat(1, window_size).reshape(-1)
    labels_away_goals = labels_away_goals.repeat(1, window_size).reshape(-1)

    home_graphs_list: list[Data] = []
    away_graphs_list: list[Data] = []
    home_features_list: list[torch.Tensor] = []
    away_features_list: list[torch.Tensor] = []

    num_home_nodes = sequence[0]['home'].num_nodes
    num_away_nodes = sequence[0]['away'].num_nodes

    for t in range(window_size):
        start_from = max(0, 0 if not product_length else t - product_length + 1)
        subseq = sequence[start_from:t + 1]

        # combine into two temporal graphs
        subseq_homo_graphs_home: list[Data] = []
        subseq_homo_graphs_away: list[Data] = []
        for data in subseq:
            data_home = Data(
                x=data['home'].x,
                edge_index=data["home", "passes_to", "home"].edge_index,
                edge_attr=data["home", "passes_to", "home"].edge_weight
            )
            subseq_homo_graphs_home.append(data_home)

            data_away = Data(
                x=data['away'].x,
                edge_index=data["away", "passes_to", "away"].edge_index,
                edge_attr=data["away", "passes_to", "away"].edge_weight
            )
            subseq_homo_graphs_away.append(data_away)

        home_graph = build_product_graph(subseq_homo_graphs_home).to(device)
        home_graphs_list.append(home_graph)

        away_graph = build_product_graph(subseq_homo_graphs_away).to(device)
        away_graphs_list.append(away_graph)

        # extract team global features
        home_features, away_features = extract_global_feature_from_match(subseq[-1], device)
        home_features_list.append(home_features)
        away_features_list.append(away_features)

    out = model(home_graphs_list, away_graphs_list, home_features_list, away_features_list,
                num_home_nodes, num_away_nodes)

    return out, labels_y, labels_home_goals, labels_away_goals


# Define hyperparameters
HYPERPARAMETERS = Hyperparameters(
    num_epochs=50,
    learning_rate=5e-4,
    weight_decay=1e-5,
    patience=7,
    goal_information=False,
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
            starting_year=HYPERPARAMETERS.starting_year,
            ending_year=HYPERPARAMETERS.ending_year,
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
            input_size=12,
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
            input_size=12,
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
    "no_goals": ExperimentConfig(
        name="no_goals",
        dataset_factory=lambda: TemporalSoccerDataset(
            root="data",
            starting_year=HYPERPARAMETERS.starting_year,
            ending_year=HYPERPARAMETERS.ending_year,
            time_interval=HYPERPARAMETERS.time_interval,
        ),
        model=NoGoalsModel(),
        forward_pass=forward_pass_no_goals_baseline,
        criterion=build_criterion(
            goal_information=HYPERPARAMETERS.goal_information,
            alpha=HYPERPARAMETERS.alpha,
            beta=HYPERPARAMETERS.beta,
        ),
        only_cpu=True,
        trainable=False,
    ),
    "grnn": ExperimentConfig(
        name="grnn",
        dataset_factory=lambda: TemporalAllPlayersSoccerDataset(
            root="data",
            starting_year=HYPERPARAMETERS.starting_year,
            ending_year=HYPERPARAMETERS.ending_year,
            time_interval=HYPERPARAMETERS.time_interval,
        ),
        model=GraphRNNModel(
            hidden_size=64,
            num_layers=3,
            goal_information=HYPERPARAMETERS.goal_information
        ),
        forward_pass=forward_pass_graph_rnn,
        criterion=build_criterion(
            goal_information=HYPERPARAMETERS.goal_information,
            alpha=HYPERPARAMETERS.alpha,
            beta=HYPERPARAMETERS.beta,
        ),
        only_cpu=True,
    ),
    "product_graphs": ExperimentConfig(
        name="product_graphs",
        dataset_factory=lambda: TemporalAllPlayersSoccerDataset(
            root="data",
            starting_year=HYPERPARAMETERS.starting_year,
            ending_year=HYPERPARAMETERS.ending_year,
            time_interval=HYPERPARAMETERS.time_interval,
        ),
        model=ProductGraphsModel(
            hidden_size=32,
            num_layers=16,
            only_last=True,
            goal_information=HYPERPARAMETERS.goal_information
        ),
        forward_pass=forward_pass_product_graphs,
        criterion=build_criterion(
            goal_information=HYPERPARAMETERS.goal_information,
            alpha=HYPERPARAMETERS.alpha,
            beta=HYPERPARAMETERS.beta,
        ),
        only_cpu=True,
    ),
    "moving_product_graphs": ExperimentConfig(
        name="moving_product_graphs",
        dataset_factory=lambda: TemporalAllPlayersSoccerDataset(
            root="data",
            starting_year=HYPERPARAMETERS.starting_year,
            ending_year=HYPERPARAMETERS.ending_year,
            time_interval=HYPERPARAMETERS.time_interval,
        ),
        model=ProductGraphsModel(
            hidden_size=64,
            num_layers=8,
            only_last=False,
            goal_information=HYPERPARAMETERS.goal_information
        ),
        forward_pass=forward_pass_product_graphs,
        criterion=build_criterion(
            goal_information=HYPERPARAMETERS.goal_information,
            alpha=HYPERPARAMETERS.alpha,
            beta=HYPERPARAMETERS.beta,
        ),
        only_cpu=True,
    )
}
