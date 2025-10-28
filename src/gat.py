import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import ELU, RNN, Linear
from torch_geometric.nn import GATConv, global_mean_pool
from tqdm import tqdm

from dataloader_paired import CumulativeSoccerDataset


class Classifier(torch.nn.Module):
    def __init__(self, input_size=64, hidden_size=16, num_classes=3):
        super().__init__()
        self.lin1 = Linear(input_size, hidden_size)
        self.lin2 = Linear(hidden_size, num_classes)

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = self.lin2(x)

        return x
    
class GoalPredictor(torch.nn.Module):
    def __init__(self, input_size=64, hidden_size=16, num_classes=1):
        super().__init__()
        self.lin1 = Linear(input_size, hidden_size)
        self.lin2 = Linear(hidden_size, num_classes)

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return torch.exp(x)


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


# 1 GAT Shared across all windows or 1 GAT per window?
# Concatenate x1 and x2 and feed that to an RNN, or have 2 RNNs (one for x1 and one for x2) and then concatenate their outputs?
class DisjointModel(torch.nn.Module):
    def __init__(
        self,
        window_size=5,
        hidden_dim=32,
        input_size=4,       # [player_id, pos_x, pos_y, is_valid]
        N1=128,
        N2=128,
        N3=64,
        N4=64,
        N5=16,
        L=6,
        num_classes=3,
        goal_information=False
    ):
        super().__init__()
        self.window_size = window_size
        self.N4 = N4
        self.goal_information = goal_information

        self.gat = GAT(input_size, N1, N2, N3, N4, L)
        self.rnn = torch.nn.GRU(
            2 * N4, hidden_dim, batch_first=True 
        )
        self.classifier = Classifier(hidden_dim, N5, num_classes)
        if goal_information:
            self.goal_home_predicter = GoalPredictor(hidden_dim, N5, 1)
            self.goal_away_predicter = GoalPredictor(hidden_dim, N5, 1)

    def forward(self, x1, x2, edge_index1, edge_index2, edge_weight1, edge_weight2, batch1, batch2, x_norm2_1, x_norm2_2, batch_size, window_size):
        device = next(self.parameters()).device
        
        # outputs: [batch_size, window_size, 2 * N4]
        outputs = torch.zeros(batch_size, window_size, 2 * self.N4, device=device)


        # Process each timestep of all windows
        for timestep_idx in range(window_size):
            x1_t = x1[timestep_idx]          # [total_nodes_home, 4]
            x2_t = x2[timestep_idx]          # [total_nodes_away, 4]
            edge1_t = edge_index1[timestep_idx]  # [2, total_edges_home]
            edge2_t = edge_index2[timestep_idx]  # [2, total_edges_away]
            edgew1_t = edge_weight1[timestep_idx] # [total_edges_home]
            edgew2_t = edge_weight2[timestep_idx] # [total_edges_away]
            batch1_t = batch1[timestep_idx]  # [total_nodes_home]
            batch2_t = batch2[timestep_idx]  # [total_nodes_away]
            
            x_norm1_t = x_norm2_1[timestep_idx]  # [batch_size, 6]
            x_norm2_t = x_norm2_2[timestep_idx]  # [batch_size, 6]
            
            # Verificar NaN en inputs
            if torch.isnan(x1_t).any() or torch.isnan(x_norm1_t).any():
                print("⚠️ NaN detected in INPUTS!")
                return torch.randn(batch_size, 3, device=device)  # Return random outputs para continuar

            x1_out, x2_out = self.gat(
                x1_t, x2_t, 
                edge1_t, edge2_t, 
                batch1_t, batch2_t, 
                x_norm1_t, x_norm2_t,
                edge_col1 = edgew1_t, edge_col2 = edgew2_t
            )

            # Verificar NaN en outputs del GAT
            if torch.isnan(x1_out).any() or torch.isnan(x2_out).any():
                print("⚠️ NaN detected in GAT outputs!")
                print(f"  x1_out: {x1_out}")
                print(f"  x2_out: {x2_out}")
                return torch.randn(batch_size, 3, device=device)
            
            combined = torch.cat([x1_out, x2_out], dim=-1)  # [batch_size, 2 * N4]
            outputs[:, timestep_idx, :] = combined
        
        # Verificar NaN antes del RNN
        if torch.isnan(outputs).any():
            print("⚠️ NaN detected in RNN inputs!")
            return torch.randn(batch_size, 3, device=device)

        rnn_out, hidden = self.rnn(outputs)
        
        # Verificar NaN después del RNN
        if torch.isnan(hidden).any():
            print("⚠️ NaN detected in RNN hidden states!")
            return torch.randn(batch_size, 3, device=device)
        
        if self.goal_information:
            return {"class_logits": self.classifier(hidden[-1]),
                    "home_goals_pred": self.goal_home_predicter(hidden[-1]),
                    "away_goals_pred": self.goal_home_predicter(hidden[-1])
            }
        else:
            return self.classifier(hidden[-1])


def main():
    dataset = CumulativeSoccerDataset(
        root="data", starting_year=2015, ending_year=2016, time_interval=30
    )
    model = SpatialModel(input_size=1, L=0)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0)

    model.train()
    num_epochs = 100

    for epoch in tqdm(range(num_epochs)):
        epoch_loss = 0.0
        for i in tqdm(range(200)):
            data = dataset[i]
            if data.home_x.size(0) == data.away_x.size(0):
                batch = torch.zeros(
                    data.home_x.size(0), dtype=torch.long, device=data.home_x.device
                )
                optimizer.zero_grad()
                x = model(
                    data.home_x,
                    data.away_x,
                    data.home_edge_index,
                    data.away_edge_index,
                    batch,
                    None,
                    None,
                    None,
                )

                loss = criterion(x.squeeze(), data.final_result)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss / len(dataset):.4f}")


if __name__ == "__main__":
    main()
