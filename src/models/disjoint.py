import torch

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
        x_norm2_1,
        x_norm2_2,
        window_size,
    ):
        device = next(self.parameters()).device

        # outputs: [batch_size, window_size, 2 * N4]
        outputs = torch.zeros(window_size, 2 * self.N4, device=device)

        # Process each timestep of all windows
        for timestep_idx in range(window_size):
            x1_t = x1[timestep_idx]  # [total_nodes_home, 4]
            x2_t = x2[timestep_idx]  # [total_nodes_away, 4]
            edge1_t = edge_index1[timestep_idx]  # [2, total_edges_home]
            edge2_t = edge_index2[timestep_idx]  # [2, total_edges_away]
            edgew1_t = edge_weight1[timestep_idx]  # [total_edges_home]
            edgew2_t = edge_weight2[timestep_idx]  # [total_edges_away]

            x_norm1_t = x_norm2_1[timestep_idx]  # [batch_size, 6]
            x_norm2_t = x_norm2_2[timestep_idx]  # [batch_size, 6]

            x1_out, x2_out = self.gat(
                x1_t,
                x2_t,
                edge1_t,
                edge2_t,
                x_norm1_t,
                x_norm2_t,
                edge_col1=edgew1_t,
                edge_col2=edgew2_t,
            )

            combined = torch.cat([x1_out, x2_out])  # [batch_size, 2 * N4]
            outputs[timestep_idx, :] = combined

        # WATCH OUT IF WE USE BATCH SIZE
        rnn_out, hidden = self.rnn(outputs.unsqueeze(0))
        rnn_out = rnn_out.squeeze(0)

        if self.goal_information:
            return {
                "class_logits": self.classifier(rnn_out),
                "home_goals_pred": self.goal_home_predicter(rnn_out),
                "away_goals_pred": self.goal_away_predicter(rnn_out),
            }
        else:
            return {"class_logits": self.classifier(rnn_out)}
