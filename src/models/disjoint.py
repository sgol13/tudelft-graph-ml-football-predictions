import torch
import torch.nn.functional as F
from torch.nn import Linear

from models.gat import GAT, Classifier, GoalPredictor


# 1 GAT Shared across all windows or 1 GAT per window?
# Concatenate x1 and x2 and feed that to an RNN, or have 2 RNNs (one for x1 and one for x2) and then concatenate their outputs?
class DisjointModel(torch.nn.Module):
    def __init__(
        self,
        hidden_dim=32,
        input_size=4,  # [player_id, pos_x, pos_y, is_valid]
        N1=128,
        N2=128,
        N3=64,
        N4=64,
        N5=16,
        L=6,
        num_classes=3,
        goal_information=False,
    ):
        super().__init__()
        self.N4 = N4
        self.goal_information = goal_information

        self.gat = GAT(input_size, N1, N2, N3, N4, L)
        self.rnn = torch.nn.GRU(2 * N4, hidden_dim, batch_first=True)
        self.classifier = Classifier(hidden_dim, N5, num_classes)
        if goal_information:
            self.goal_home_predicter = GoalPredictor(hidden_dim, N5, 1)
            self.goal_away_predicter = GoalPredictor(hidden_dim, N5, 1)

    def forward(
        self,
        x1,
        x2,
        edge_index1,
        edge_index2,
        edge_weight1,
        edge_weight2,
        batch1,
        batch2,
        x_norm2_1,
        x_norm2_2,
        batch_size,
        window_size,
    ):
        device = next(self.parameters()).device

        # outputs: [batch_size, window_size, 2 * N4]
        outputs = torch.zeros(batch_size, window_size, 2 * self.N4, device=device)

        # Process each timestep of all windows
        for timestep_idx in range(window_size):
            x1_t = x1[timestep_idx]  # [total_nodes_home, 4]
            x2_t = x2[timestep_idx]  # [total_nodes_away, 4]
            edge1_t = edge_index1[timestep_idx]  # [2, total_edges_home]
            edge2_t = edge_index2[timestep_idx]  # [2, total_edges_away]
            edgew1_t = edge_weight1[timestep_idx]  # [total_edges_home]
            edgew2_t = edge_weight2[timestep_idx]  # [total_edges_away]
            batch1_t = batch1[timestep_idx]  # [total_nodes_home]
            batch2_t = batch2[timestep_idx]  # [total_nodes_away]

            x_norm1_t = x_norm2_1[timestep_idx]  # [batch_size, 6]
            x_norm2_t = x_norm2_2[timestep_idx]  # [batch_size, 6]

            # Verificar NaN en inputs
            if torch.isnan(x1_t).any() or torch.isnan(x_norm1_t).any():
                print("⚠️ NaN detected in INPUTS!")
                return torch.randn(
                    batch_size, 3, device=device
                )  # Return random outputs para continuar

            x1_out, x2_out = self.gat(
                x1_t,
                x2_t,
                edge1_t,
                edge2_t,
                batch1_t,
                batch2_t,
                x_norm1_t,
                x_norm2_t,
                edge_col1=edgew1_t,
                edge_col2=edgew2_t,
            )

            # Verify there are no NaNs in the outputs
            if torch.isnan(x1_out).any() or torch.isnan(x2_out).any():
                print("⚠️ NaN detected in GAT outputs!")
                print(f"  x1_out: {x1_out}")
                print(f"  x2_out: {x2_out}")
                return torch.randn(batch_size, 3, device=device)

            combined = torch.cat([x1_out, x2_out], dim=-1)  # [batch_size, 2 * N4]
            outputs[:, timestep_idx, :] = combined

        # Verify there are no NaNs in the outputs
        if torch.isnan(outputs).any():
            print("⚠️ NaN detected in RNN outputs!")
            return torch.randn(batch_size, 3, device=device)

        _, hidden = self.rnn(outputs)

        # Verify there are no NaNs in the hidden states
        if torch.isnan(hidden).any():
            print("⚠️ NaN detected in RNN hidden states!")
            return torch.randn(batch_size, 3, device=device)

        if self.goal_information:
            return {
                "class_logits": self.classifier(hidden[-1]),
                "home_goals_pred": self.goal_home_predicter(hidden[-1]),
                "away_goals_pred": self.goal_away_predicter(hidden[-1]),
            }
        else:
            return {"class_logits" : self.classifier(hidden[-1])}
