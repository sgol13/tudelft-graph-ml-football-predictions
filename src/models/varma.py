import torch
import torch.nn as nn


class VARMABaseline(nn.Module):
    def __init__(
        self,
        input_size=6,
        hidden_size=64,
        p=2,
        q=1,
        num_classes=3,
        goal_information=False,
    ):
        super(VARMABaseline, self).__init__()

        self.p = p
        self.q = q
        self.input_size = input_size
        self.hidden_size = hidden_size

        # AR (autoregressive) weights: one matrix per lag
        self.AR = nn.ModuleList([nn.Linear(input_size, hidden_size) for _ in range(p)])
        # MA (moving average) weights: one matrix per lag
        self.MA = nn.ModuleList([nn.Linear(input_size, hidden_size) for _ in range(q)])

        # Combine to output
        self.fc_out = nn.Linear(hidden_size, num_classes)

        if goal_information:
            self.goal_home_predicter = nn.Linear(hidden_size, 1)
            self.goal_away_predicter = nn.Linear(hidden_size, 1)
        self.goal_information = goal_information

    def forward(self, Y, e=None):
        # Y: tensor of shape [batch_size, seq_len, input_size]
        # e: residuals (same shape as Y)

        batch_size, seq_len, input_size = Y.shape

        # Start with zeros
        hidden = torch.zeros(batch_size, self.hidden_size, device=Y.device)

        # --- Handle short sequences safely ---
        effective_p = min(self.p, seq_len)
        effective_q = min(self.q, e.shape[1] if e is not None else 0)

        # AR component
        for i in range(1, effective_p + 1):
            hidden += self.AR[i - 1](Y[:, -i, :])  # use last p steps

        # MA component
        if e is not None and effective_q > 0:
            for j in range(1, effective_q + 1):
                hidden += self.MA[j - 1](e[:, -j, :])

        # Nonlinearity
        hidden = torch.tanh(hidden)

        # Output
        if self.goal_information:
            return {
                "class_logits": self.fc_out(hidden),
                "home_goals_pred": self.goal_home_predicter(hidden),
                "away_goals_pred": self.goal_away_predicter(hidden),
            }
        else:
            return {"class_logits": self.fc_out(hidden)}  # [batch, output_size]
