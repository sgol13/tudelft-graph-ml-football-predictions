import torch
import torch.nn as nn


class SimpleRNNModel(nn.Module):
    def __init__(
        self, input_size, hidden_size, num_layers, output_size, goal_information
    ):
        super(SimpleRNNModel, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,  # input shape: (batch, seq, features)
        )

        self.fc = nn.Linear(hidden_size, output_size)
        if goal_information:
            self.goal_home_predicter = nn.Linear(hidden_size, 1)
            self.goal_away_predicter = nn.Linear(hidden_size, 1)
        self.goal_information = goal_information

    def forward(self, x):
        """
        x: [batch, seq_len, input_size]
        returns: [batch, output_size]
        """

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, hn = self.rnn(x, h0)  # out: [batch, seq_len, hidden_size]
        out = out.squeeze(0)

        # out = out[:, -1, :]  # [batch, hidden_size]
        if self.goal_information:
            return {
                "class_logits": self.fc(out),
                "home_goals_pred": self.goal_home_predicter(out),
                "away_goals_pred": self.goal_away_predicter(out),
            }
        else:
            return {"class_logits": self.fc(out)}  # [batch, output_size]
