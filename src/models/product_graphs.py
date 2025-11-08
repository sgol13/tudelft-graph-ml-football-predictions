import torch.nn as nn
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

from models.gat import GoalPredictor


class ProductGraphsModel(nn.Module):
    def __init__(self, hidden_size: int = 32, num_layers: int = 5, num_classes: int = 3, num_player_features: int = 4,
                 num_global_features: int = 6, only_last: bool = False, goal_information: bool = False):
        super(ProductGraphsModel, self).__init__()

        self.goal_information: bool = goal_information
        self.only_last: bool = only_last

        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(num_player_features, hidden_size))
        for l in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_size, hidden_size))

        grnn_output_size = 2 * (hidden_size + num_global_features)
        self.fc = nn.Linear(grnn_output_size, num_classes)

        if goal_information:
            self.goal_home_predictor = GoalPredictor(grnn_output_size, hidden_size, 1)
            self.goal_away_predictor = GoalPredictor(grnn_output_size, hidden_size, 1)

    def forward(self, home_graphs_list: list[Data], away_graphs_list: list[Data],
                home_features_list: list[torch.Tensor], away_features_list: list[torch.Tensor],
                num_home_nodes: int, num_away_nodes: int):

        output_logits = []
        home_goals_pred = []
        away_goals_pred = []

        for home_graph, away_graph, home_features, away_features in zip(home_graphs_list, away_graphs_list,
                                                                        home_features_list, away_features_list):

            y_home = self.forward_input_graph(home_graph)
            y_away = self.forward_input_graph(away_graph)

            if self.only_last:
                y_home_pooled = y_home[:num_home_nodes].mean(dim=0)
                y_away_pooled = y_away[:num_away_nodes].mean(dim=0)
            else:
                y_home_pooled = y_home.mean(dim=0)
                y_away_pooled = y_away.mean(dim=0)

            combined_features = torch.cat([y_home_pooled, home_features, y_away_pooled, away_features])

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

    def forward_input_graph(self, graph: Data) -> torch.Tensor:
        x = graph.x
        for conv in self.convs:
            x = conv(x, edge_index=graph.edge_index, edge_weight=graph.edge_attr)
            x = torch.relu(x)

        return x
