import torch.nn as nn
import torch
from typing import List, Dict, Any


def build_criterion(goal_information: bool,  alpha: float = 1.0, beta: float = 0.5):
    """Return criterion callable with consistent signature."""

    if not goal_information:
        ce = nn.CrossEntropyLoss()
        return lambda outputs, labels_y, labels_home_goals, labels_away_goals: ce(outputs["class_logits"], labels_y)


    ce = nn.CrossEntropyLoss()
    poisson = nn.PoissonNLLLoss(log_input=False)
    eps = 1e-6


    def compute_loss(outputs: Dict[str, torch.Tensor], labels_y: torch.Tensor, labels_home_goals: torch.Tensor, labels_away_goals: torch.Tensor):
        loss_cls = ce(outputs["class_logits"], labels_y)
        home_pred = outputs["home_goals_pred"].clamp_min(eps).squeeze()
        away_pred = outputs["away_goals_pred"].clamp_min(eps).squeeze()
        loss_home = poisson(home_pred, labels_home_goals.float().squeeze())
        loss_away = poisson(away_pred, labels_away_goals.float().squeeze())
        loss_goals = 0.5 * (loss_home + loss_away)
        return alpha * loss_cls + beta * loss_goals

    return compute_loss