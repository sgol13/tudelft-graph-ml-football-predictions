import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import GCN

from src.models.gat import GoalPredictor


class GraphRNNModel(nn.Module):
    def __init__(self, hidden_size: int = 16, num_layers: int = 1, num_classes: int = 3, num_player_features: int = 4, num_global_features: int = 6, goal_information: bool = False):
        super(GraphRNNModel, self).__init__()

        self.hidden_size: int = hidden_size
        self.num_classes: int = num_classes
        self.goal_information: bool = goal_information

        self.A = GCN(in_channels=num_player_features, hidden_channels=hidden_size, num_layers=num_layers)
        self.B = GCN(in_channels=hidden_size, hidden_channels=hidden_size, num_layers=num_layers)
        self.C = GCN(in_channels=hidden_size, hidden_channels=hidden_size, num_layers=num_layers)

        grnn_output_size = 2 * (hidden_size + num_global_features)
        self.fc = nn.Linear(grnn_output_size, num_classes)

        if goal_information:
            self.goal_home_predictor = GoalPredictor(grnn_output_size, hidden_size, 1)
            self.goal_away_predictor = GoalPredictor(grnn_output_size, hidden_size, 1)

    def forward(self, home_graphs: list[Data], away_graphs: list[Data], home_features: list[torch.Tensor],
                away_features: list[torch.Tensor], window_size: int):

        device = next(self.parameters()).device
        num_home_nodes = home_graphs[0].num_nodes
        num_away_nodes = away_graphs[0].num_nodes

        Z_home = torch.zeros(num_home_nodes, self.hidden_size, device=device)
        Z_away = torch.zeros(num_away_nodes, self.hidden_size, device=device)

        output_logits = []
        home_goals_pred = []
        away_goals_pred = []

        for t in range(window_size):
            Z_home, Y_home = self.forward_input_graph(home_graphs[t], Z_home)

            Z_away, Y_away = self.forward_input_graph(away_graphs[t], Z_away)

            y_home_pooled = Y_home.mean(dim=0)
            y_away_pooled = Y_away.mean(dim=0)

            combined_features = torch.cat([y_home_pooled, home_features[t], y_away_pooled, away_features[t]])

            logits = self.fc(combined_features)
            output_logits.append(logits)

            if self.goal_information:
                home_goals = self.goal_home_predictor(combined_features)
                away_goals = self.goal_away_predictor(combined_features)

                home_goals_pred.append(home_goals)
                away_goals_pred.append(away_goals)

        if self.goal_information:
            return {
                "class_logits": torch.stack(output_logits),
                "home_goals_pred": torch.stack(home_goals_pred),
                "away_goals_pred": torch.stack(away_goals_pred),
            }
        else:
            return {"class_logits": torch.stack(output_logits)}

    def forward_input_graph(self, input: Data, Z_previous: torch.Tensor) -> torch.Tensor:
        Z = nn.functional.relu(
            self.A(x=input.x, edge_index=input.edge_index, edge_weight=input.edge_attr) +
            self.B(Z_previous, edge_index=input.edge_index, edge_weight=input.edge_attr)
        )
        Y = nn.functional.relu(self.C(Z, edge_index=input.edge_index, edge_weight=input.edge_attr))
        return Z, Y
