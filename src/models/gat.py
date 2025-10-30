import torch
import torch.nn.functional as F
from torch.nn import ELU, Linear
from torch_geometric.nn import GATConv, global_mean_pool


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
        batch1,
        batch2,
        x_norm2_1,
        x_norm2_2,
        edge_col1=None,
        edge_col2=None,
    ):
        # 1. Obtain node embeddings
        x1 = self.elu(self.conv1(x1, edge_index1, edge_col1))
        x1 = F.dropout(x1, p=0.2, training=self.training)

        x1 = self.elu(self.conv2(x1, edge_index1, edge_col1))
        x1 = F.dropout(x1, p=0.2, training=self.training)

        x1 = self.elu(self.conv3(x1, edge_index1, edge_col1))
        x1 = F.dropout(x1, p=0.2, training=self.training)

        x2 = self.elu(self.conv1(x2, edge_index2, edge_col2))
        x2 = F.dropout(x2, p=0.2, training=self.training)

        x2 = self.elu(self.conv2(x2, edge_index2, edge_col2))
        x2 = F.dropout(x2, p=0.2, training=self.training)

        x2 = self.elu(self.conv3(x2, edge_index2, edge_col2))
        x2 = F.dropout(x2, p=0.2, training=self.training)

        # 2. Readout layer
        # Two batches since graphs don't necessarily have the same number of nodes
        x1 = global_mean_pool(x1, batch1)
        x2 = global_mean_pool(x2, batch2)

        if x_norm2_1 is not None:
            x1 = torch.cat((x1, x_norm2_1), dim=1)
        if x_norm2_2 is not None:
            x2 = torch.cat((x2, x_norm2_2), dim=1)

        x1 = self.lin(x1)
        x2 = self.lin(x2)

        return x1, x2


class SpatialModel(torch.nn.Module):
    def __init__(
        self, input_size=7, N1=128, N2=128, N3=64, N4=64, N5=16, L=16, num_classes=3
    ):
        super().__init__()
        self.gat = GAT(input_size, N1, N2, N3, N4, L)
        # After GAT, each graph has dimension N4, and we concatenate two graphs
        self.classifier = Classifier(2 * N4, N5, num_classes)

    def forward(
        self,
        x1,
        x2,
        edge_index1,
        edge_index2,
        batch1,
        batch2,
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
            batch1,
            batch2,
            x_norm2_1,
            x_norm2_2,
            edge_col1,
            edge_col2,
        )
        x = torch.cat((x1, x2), dim=1)  # x has both graph embeddings
        # For the Disjoint Model, we want this x as output of each GAT

        x = self.classifier(x)
        return x
