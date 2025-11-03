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
        """
        Y: [batch_size, seq_len, input_size]
        e: residuals [batch_size, seq_len, input_size] or None
        Returns:
            dict with keys:
                - "class_logits": [seq_len, num_classes] if batch_size=1 else [batch_size, seq_len, num_classes]
                - optionally "home_goals_pred", "away_goals_pred": same convention
        """
        batch_size, seq_len, input_size = Y.shape

        class_logits = []
        if self.goal_information:
            home_goals_pred = []
            away_goals_pred = []

        for t in range(seq_len):
            ar_start = max(0, t - self.p + 1)
            ar_window = Y[:, ar_start : t + 1, :]

            hidden = torch.zeros(batch_size, self.hidden_size, device=Y.device)

            # AR contribution
            for i, ar_step in enumerate(reversed(range(ar_window.size(1)))):
                hidden += self.AR[i](ar_window[:, ar_step, :])

            # MA contribution
            if e is not None:
                ma_start = max(0, t - self.q + 1)
                ma_window = e[:, ma_start : t + 1, :]
                for j, ma_step in enumerate(reversed(range(ma_window.size(1)))):
                    hidden += self.MA[j](ma_window[:, ma_step, :])

            hidden = torch.tanh(hidden)

            class_logits.append(self.fc_out(hidden))
            if self.goal_information:
                home_goals_pred.append(self.goal_home_predicter(hidden))
                away_goals_pred.append(self.goal_away_predicter(hidden))

        # Stack over time dimension
        class_logits = torch.stack(class_logits, dim=1)  # [batch, seq_len, num_classes]
        if self.goal_information:
            home_goals_pred = torch.stack(home_goals_pred, dim=1)
            away_goals_pred = torch.stack(away_goals_pred, dim=1)

        # If batch_size==1, squeeze batch dim
        if batch_size == 1:
            class_logits = class_logits.squeeze(0)  # [seq_len, num_classes]
            if self.goal_information:
                home_goals_pred = home_goals_pred.squeeze(0)
                away_goals_pred = away_goals_pred.squeeze(0)

        output_dict = {"class_logits": class_logits}
        if self.goal_information:
            output_dict["home_goals_pred"] = home_goals_pred
            output_dict["away_goals_pred"] = away_goals_pred

        return output_dict
