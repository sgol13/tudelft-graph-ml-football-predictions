import torch
import torch.nn.functional as F
from torch.nn import ELU, Linear
from torch_geometric.nn import GATConv


class GoalPredictor(torch.nn.Module):
    def __init__(self, input_size=64, hidden_size=16, num_classes=1):
        super().__init__()
        self.lin1 = Linear(input_size, hidden_size)
        self.lin2 = Linear(hidden_size, num_classes)

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return torch.exp(x)


class Classifier(torch.nn.Module):
    def __init__(self, input_size=64, hidden_size=16, num_classes=3):
        super().__init__()
        self.lin1 = Linear(input_size, hidden_size)
        self.lin2 = Linear(hidden_size, num_classes)

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = self.lin2(x)

        return x


class GAT(torch.nn.Module):
    def __init__(self, input_size=7, N1=128, N2=128, N3=64, N4=64, L=16, edge_dim=1):
        super().__init__()
        self.conv1 = GATConv(input_size, N1, edge_dim=edge_dim)
        self.conv2 = GATConv(N1, N2, edge_dim=edge_dim)
        self.conv3 = GATConv(N2, N3, edge_dim=edge_dim)

        self.lin = Linear(N3 + L, N4)
        self.elu = ELU()

    def forward(
        self,
        x1,
        x2,
        edge_index1,
        edge_index2,
        x_norm2_1,
        x_norm2_2,
        edge_col1=None,
        edge_col2=None,
    ):
        def process_graph(x, edge_index, edge_attr):
            x = self.elu(self.conv1(x, edge_index, edge_attr))
            x = F.dropout(x, p=0.2, training=self.training)
            x = self.elu(self.conv2(x, edge_index, edge_attr))
            x = F.dropout(x, p=0.2, training=self.training)
            x = self.elu(self.conv3(x, edge_index, edge_attr))
            x = F.dropout(x, p=0.2, training=self.training)
            x = x.mean(dim=0)
            return x

        x1 = process_graph(x1, edge_index1, edge_col1)
        x2 = process_graph(x2, edge_index2, edge_col2)

        if x_norm2_1 is not None:
            x1 = torch.cat((x1, x_norm2_1))
        if x_norm2_2 is not None:
            x2 = torch.cat((x2, x_norm2_2))

        x1 = self.lin(x1)
        x2 = self.lin(x2)

        return x1, x2


class SpatialModel(torch.nn.Module):
    def __init__(
        self,
        input_size=7,
        N1=128,
        N2=128,
        N3=64,
        N4=64,
        N5=16,
        L=16,
        num_classes=3,
        goal_information=False,
    ):
        super().__init__()
        self.gat = GAT(input_size, N1, N2, N3, N4, L)
        # After GAT, each graph has dimension N4, and we concatenate two graphs
        self.classifier = Classifier(2 * N4, N5, num_classes)
        if goal_information:
            self.goal_home_predicter = GoalPredictor(2 * N4, N5, 1)
            self.goal_away_predicter = GoalPredictor(2 * N4, N5, 1)
        self.goal_information = goal_information

    def forward(
        self,
        x1,
        x2,
        edge_index1,
        edge_index2,
        x_norm2_1,
        x_norm2_2,
        edge_col1=None,
        edge_col2=None,
    ):
        x1, x2 = self.gat(
            x1,
            x2,
            edge_index1,
            edge_index2,
            x_norm2_1,
            x_norm2_2,
            edge_col1,
            edge_col2,
        )
        x = torch.cat((x1, x2))
        # For the Disjoint Model, we want this x as output of each GAT

        if self.goal_information:
            return {
                "class_logits": self.classifier(x).unsqueeze(0),
                "home_goals_pred": self.goal_home_predicter(x).unsqueeze(0),
                "away_goals_pred": self.goal_away_predicter(x).unsqueeze(0),
            }
        else:
            return {"class_logits": self.classifier(x).unsqueeze(0)}
