import torch.nn as nn
import torch

from dataloader_paired import TemporalSequence


class NoGoalsModel(nn.Module):
    def __init__(self):
        super(NoGoalsModel, self).__init__()
        self.param = nn.Parameter(torch.zeros(1))

    def forward(self, entry: TemporalSequence, window_size: int):

        device = next(self.parameters()).device
        results = []
        home_goals = []
        away_goals = []

        for window in entry.hetero_data_sequence[:window_size]:
            # always predict true final result
            # home = entry.final_home_goals
            # away = entry.final_away_goals

            # use current result as the prediction
            home = window.current_home_goals
            away = window.current_away_goals

            if home > away:
                r = torch.tensor([1.0, 0.0, 0.0], device=device)
            elif home < away:
                r = torch.tensor([0.0, 0.0, 1.0], device=device)
            else:
                r = torch.tensor([0.0, 1.0, 0.0], device=device)

            results.append(r)
            home_goals.append(home)
            away_goals.append(away)

        return {
            'class_logits': torch.stack(results),
            'home_goals_pred': torch.tensor(home_goals, device=device),
            'away_goals_pred': torch.tensor(away_goals, device=device),
        }
