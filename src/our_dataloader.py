import pickle
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from numpy import ndarray
from pandas import DataFrame
from torch_geometric.data import Data, Dataset
from tqdm import tqdm


def count_passes(
    team_events: pd.DataFrame,
) -> tuple[Counter[tuple[int, int]], ndarray[str]]:
    """Count successful passes within a team over given events."""
    if team_events.empty:
        return Counter(), np.array([]).astype(str)

    team_players = team_events["player"].dropna().unique().astype(str)
    player_to_idx = {p: i for i, p in enumerate(team_players)}

    rows = team_events[["type", "outcome_type", "player"]].to_numpy()
    passes: list[tuple[int, int]] = []
    pass_from = None

    for event_type, outcome_type, player in rows:
        if event_type == "Pass" and outcome_type == "Successful":
            if player not in player_to_idx:
                continue
            if pass_from is not None and pass_from in player_to_idx:
                passes.append((player_to_idx[pass_from], player_to_idx[player]))
        pass_from = player

    pass_counts = Counter(passes)
    return pass_counts, team_players


def build_graph(
    pass_counts: Counter[tuple[int, int]],
    team_players: list[str],
    team: str,
    time_range: tuple[int, int],
) -> Data:
    if not pass_counts:
        raise ValueError("No passes found.")

    edge_index = (
        torch.tensor(list(pass_counts.keys()), dtype=torch.long).t().contiguous()
    )
    edge_weight = torch.tensor(list(pass_counts.values()), dtype=torch.float)

    # Node features (placeholder: just node indices)
    x = torch.arange(len(team_players), dtype=torch.float).unsqueeze(1)

    data = Data(x=x, edge_index=edge_index, edge_weight=edge_weight)
    data.team = team
    data.time_range = time_range
    data.players = team_players  # This is a numpy array, and I think it should be a Torch Tensor or a List (for compatibility)

    return data


def build_team_graphs(
    events, time_interval=5, cumulative=False
) -> dict[str, list[Data]]:
    max_minute = events["minute"].max()
    teams = events["team"].dropna().unique()

    graphs = defaultdict(list)
    cumulative_pass_counts = defaultdict(Counter)
    cumulative_team_players = defaultdict(set)

    # Iterate over time intervals
    for start_minute in range(0, max_minute, time_interval):
        end_minute = min(start_minute + time_interval, max_minute + 1)
        events_in_interval = events[
            (events["minute"] >= start_minute)
            & (events["minute"] < end_minute)
            & (~events["type"].isin(["Start", "End", "FormationSet"]))
        ]

        for team in teams:
            team_events = events_in_interval[events_in_interval["team"] == team]
            pass_counts, team_players = count_passes(team_events)
            if not pass_counts:
                continue

            if cumulative:
                # Update cumulative stats
                for (i, j), count in pass_counts.items():
                    p_from = team_players[i]
                    p_to = team_players[j]
                    cumulative_pass_counts[team][(p_from, p_to)] += count
                cumulative_team_players[team].update(team_players)
                team_players = sorted(list(cumulative_team_players[team]))
                player_to_idx = {p: i for i, p in enumerate(team_players)}
                pass_counts = Counter(
                    {
                        (player_to_idx[i], player_to_idx[j]): count
                        for _, ((i, j), count) in cumulative_pass_counts.items()
                        if i in player_to_idx and j in player_to_idx
                    }
                )

            # Create a segmented graph
            data = build_graph(
                pass_counts, team_players, team, (start_minute, end_minute)
            )
            if data is not None:
                graphs[team].append(data)

    return graphs


class SoccerDataset(Dataset):
    def __init__(
        self,
        root,
        pre_filter=None,
        pre_transform=None,
        time_interval=5,
        starting_year=2015,
        ending_year=2024,
    ):
        self.time_interval = time_interval
        self.starting_year = starting_year
        self.ending_year = ending_year
        super().__init__(root=root, pre_filter=pre_filter, pre_transform=pre_transform)

    @property
    def raw_file_names(self):
        return [
            f"epl_{year}.pkl"
            for year in range(self.starting_year, self.ending_year + 1)
        ]

    @property
    def processed_file_names(self):
        # List all processed .pt files that already exist.
        processed = sorted(Path(self.processed_dir).glob("data_*.pt"))
        return [p.name for p in processed] if processed else ["data_0.pt"]

    def _process_match(self, events: DataFrame) -> dict[str, list[Data]]:
        # This is now abstract and should be implemented by subclasses.
        raise NotImplementedError()

    def process(self):
        idx = 0
        for raw_path in tqdm(
            self.raw_paths, desc="Processing raw data files", unit="file"
        ):
            with open(raw_path, "rb") as f:
                raw_data = pickle.load(f)

                for raw_match in tqdm(
                    raw_data,
                    desc=f"Processing matches from {Path(raw_path).name}",
                    leave=False,
                    unit="match",
                ):
                    events = raw_match["events"]
                    graphs_by_team = self._process_match(events)

                    for team, graphs in graphs_by_team.items():
                        for graph in graphs:
                            torch.save(
                                graph,
                                Path(self.processed_dir).joinpath(f"data_{idx}.pt"),
                            )
                            idx += 1

    def len(self):
        return len(list(Path(self.processed_dir).glob("data_*.pt")))

    def get(self, idx):
        path = sorted(Path(self.processed_dir).glob("data_*.pt"))[
            idx
        ]  # This sorted() here?
        return torch.load(path, weights_only=False)


class SequentialSoccerDataset(SoccerDataset):
    def __init__(
        self,
        root,
        pre_filter=None,
        pre_transform=None,
        time_interval=5,
        starting_year=2015,
        ending_year=2024,
    ):
        super().__init__(
            root=root,
            pre_filter=pre_filter,
            pre_transform=pre_transform,
            time_interval=time_interval,
            starting_year=starting_year,
            ending_year=ending_year,
        )
        # print(self)

    def _process_match(self, events: DataFrame) -> dict[str, list[Data]]:
        graphs = build_team_graphs(
            events, time_interval=self.time_interval, cumulative=False
        )
        return graphs


class CumulativeSoccerDataset(SoccerDataset):
    def __init__(
        self,
        root,
        pre_filter=None,
        pre_transform=None,
        time_interval=5,
        starting_year=2015,
        ending_year=2024,
    ):
        super().__init__(
            root=root,
            pre_filter=pre_filter,
            pre_transform=pre_transform,
            time_interval=time_interval,
            starting_year=starting_year,
            ending_year=ending_year,
        )

    def _process_match(self, events: DataFrame) -> dict[str, list[Data]]:
        graphs = build_team_graphs(
            events, time_interval=self.time_interval, cumulative=True
        )
        return graphs


cum_dataset = SequentialSoccerDataset(root="data", ending_year=2016, time_interval=30)
print(len(cum_dataset))
for i in range(100):
    print(cum_dataset[i].time_range, cum_dataset[i].team)
