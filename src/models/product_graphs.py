import torch.nn as nn
import torch
from torch_geometric.data import Data


class ProductGraphsModel(nn.Module):
    def __init__(self):
        super(ProductGraphsModel, self).__init__()

        self.param = nn.Parameter(torch.zeros(1))

    def forward(self, home_graph: Data, away_graph: Data, home_features: torch.Tensor, away_features: torch.Tensor,
                window_size: int):
        return {
            'class_logits': torch.zeros(window_size, 3),
            'home_goals_pred': torch.zeros(window_size),
            'away_goals_pred': torch.zeros(window_size),
        }
