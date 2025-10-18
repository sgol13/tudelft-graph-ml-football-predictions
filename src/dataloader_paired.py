import pickle
from collections import Counter, defaultdict
from pathlib import Path
from typing import List

import pandas as pd
import torch
from torch_geometric.data import Data, Dataset
from tqdm import tqdm


def count_passes(
    team_events: pd.DataFrame,
) -> tuple[Counter[tuple[int, int]], list[str]]:
    """Count successful passes within a team over given events."""
    if team_events.empty:
        return Counter(), []

    team_players = team_events["player"].dropna().unique().astype(str)
    player_to_idx = {p: i for i, p in enumerate(team_players)}

    rows = team_events[["type", "outcome_type", "player"]].to_numpy()
    passes = []
    pass_from = None

    for event_type, outcome_type, player in rows:
        if event_type == "Pass" and outcome_type == "Successful":
            if player not in player_to_idx:
                continue
            if pass_from is not None and pass_from in player_to_idx:
                passes.append((player_to_idx[pass_from], player_to_idx[player]))
        pass_from = player

    pass_counts = Counter(passes)
    return pass_counts, team_players.tolist()


def count_goals(
    events: pd.DataFrame, end_minute: int, home_team: str, away_team: str
) -> tuple[int, int]:
    """Count goals for home and away teams up to specified minute."""
    if events.empty or "is_goal" not in events.columns:
        return 0, 0

    # Filter events up to the current interval end minute
    events_up_to_minute = events[events["minute"] <= end_minute]

    # Count home team goals
    home_goals = events_up_to_minute[
        (events_up_to_minute["team"] == home_team) & (events_up_to_minute["is_goal"])
    ].shape[0]

    # Count away team goals
    away_goals = events_up_to_minute[
        (events_up_to_minute["team"] == away_team) & (events_up_to_minute["is_goal"])
    ].shape[0]

    return home_goals, away_goals


def get_final_result(
    events: pd.DataFrame, home_team: str, away_team: str
) -> tuple[int, int, torch.Tensor]:
    """Get final match result and encode as one-hot tensor [Home Win, Draw, Away Win]."""
    if events.empty or "is_goal" not in events.columns:
        return 0, 0, torch.tensor([0, 0, 0], dtype=torch.float)

    home_goals = events[(events["team"] == home_team) & (events["is_goal"])].shape[0]
    away_goals = events[(events["team"] == away_team) & (events["is_goal"])].shape[0]

    # Create one-hot encoding for CrossEntropyLoss
    if home_goals > away_goals:
        result_tensor = torch.tensor([1, 0, 0], dtype=torch.float)  # Home Win
    elif home_goals == away_goals:
        result_tensor = torch.tensor([0, 1, 0], dtype=torch.float)  # Draw
    else:
        result_tensor = torch.tensor([0, 0, 1], dtype=torch.float)  # Away Win

    return home_goals, away_goals, result_tensor


def build_graph(pass_counts: Counter, team_players: list) -> Data:
    """Build graph - handles empty pass cases gracefully."""
    # DISCLAIMER, IF WE EVER WANT TO ADD TEAM INFO, THIS SHOULD HAVE THE NAME OF THE TEAM and others
    if not pass_counts or len(team_players) == 0:
        # Return empty graph instead of crashing
        x = torch.zeros((max(1, len(team_players)), 1), dtype=torch.float)
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_weight = torch.zeros(0, dtype=torch.float)
    else:
        edge_index = (
            torch.tensor(list(pass_counts.keys()), dtype=torch.long).t().contiguous()
        )
        edge_weight = torch.tensor(list(pass_counts.values()), dtype=torch.float)
        x = torch.arange(len(team_players), dtype=torch.float).unsqueeze(1)

    return Data(x=x, edge_index=edge_index, edge_weight=edge_weight)


def build_team_graphs_with_goals(
    events, home_team: str, away_team: str, time_interval=5, cumulative=False
) -> List[Data]:
    """Build paired graphs with both home and away teams for each time interval."""
    if events.empty or "minute" not in events.columns:
        return []

    max_minute = events["minute"].max()

    # Get final result with one-hot encoding
    final_home_goals, final_away_goals, final_result_onehot = get_final_result(
        events, home_team, away_team
    )

    paired_graphs = []
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

        # Count goals up to current interval
        current_home_goals, current_away_goals = count_goals(
            events, end_minute, home_team, away_team
        )

        home_graph = None
        away_graph = None

        # Process HOME team
        home_events = events_in_interval[events_in_interval["team"] == home_team]
        home_pass_counts, home_players = count_passes(home_events)

        if cumulative and home_pass_counts:
            for (i, j), count in home_pass_counts.items():
                p_from = home_players[i]
                p_to = home_players[j]
                cumulative_pass_counts[home_team][(p_from, p_to)] += count
            cumulative_team_players[home_team].update(home_players)
            home_players = sorted(list(cumulative_team_players[home_team]))
            player_to_idx = {p: i for i, p in enumerate(home_players)}
            home_pass_counts = Counter(
                {
                    (player_to_idx[i], player_to_idx[j]): count
                    for (i, j), count in cumulative_pass_counts[home_team].items()
                    if i in player_to_idx and j in player_to_idx
                }
            )

        home_graph = build_graph(home_pass_counts, home_players)

        # Process AWAY team
        away_events = events_in_interval[events_in_interval["team"] == away_team]
        away_pass_counts, away_players = count_passes(away_events)

        if cumulative and away_pass_counts:
            for (i, j), count in away_pass_counts.items():
                p_from = away_players[i]
                p_to = away_players[j]
                cumulative_pass_counts[away_team][(p_from, p_to)] += count
            cumulative_team_players[away_team].update(away_players)
            away_players = sorted(list(cumulative_team_players[away_team]))
            player_to_idx = {p: i for i, p in enumerate(away_players)}
            away_pass_counts = Counter(
                {
                    (player_to_idx[i], player_to_idx[j]): count
                    for (i, j), count in cumulative_pass_counts[away_team].items()
                    if i in player_to_idx and j in player_to_idx
                }
            )

        away_graph = build_graph(away_pass_counts, away_players)

        # Create PAIRED data point with both teams
        paired_data = Data(
            # Home team graph data
            home_x=home_graph.x,
            home_edge_index=home_graph.edge_index,
            home_edge_weight=home_graph.edge_weight,
            home_players=home_players,
            # Away team graph data
            away_x=away_graph.x,
            away_edge_index=away_graph.edge_index,
            away_edge_weight=away_graph.edge_weight,
            away_players=away_players,
            # Time information
            start_minute=torch.tensor(start_minute, dtype=torch.long),
            end_minute=torch.tensor(end_minute, dtype=torch.long),
            # Goal information
            current_home_goals=current_home_goals,
            current_away_goals=current_away_goals,
            final_home_goals=final_home_goals,
            final_away_goals=final_away_goals,
            final_result=final_result_onehot,
            # Team names
            home_team=home_team,
            away_team=away_team,
        )

        paired_graphs.append(paired_data)

    return paired_graphs


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
        processed = list(Path(self.processed_dir).glob("data_*.pt"))
        return [p.name for p in processed]

    def _process_match(
        self, events: pd.DataFrame, home_team: str, away_team: str
    ) -> List[Data]:
        raise NotImplementedError()

    def process(self):
        idx = 0

        for raw_path in tqdm(self.raw_paths, desc="Processing seasons"):
            season_name = Path(raw_path).stem  # e.g., "epl_2015"

            with open(raw_path, "rb") as f:
                season_data = pickle.load(f)

            for match in tqdm(
                season_data, desc=f"Matches from {season_name}", leave=False
            ):
                events = match["events"]
                home_team = match["home_team"]
                away_team = match["away_team"]
                match_id = match.get("game_id", f"{season_name}_{idx}")

                # Get paired graphs for this match
                paired_graphs = self._process_match(events, home_team, away_team)

                for paired_data in paired_graphs:
                    # Store rich context information
                    paired_data.season = season_name
                    paired_data.match_id = match_id
                    paired_data.data_id = idx

                    if self.pre_filter is not None and not self.pre_filter(paired_data):
                        continue
                    if self.pre_transform is not None:
                        paired_data = self.pre_transform(paired_data)

                    torch.save(paired_data, Path(self.processed_dir) / f"data_{idx}.pt")
                    idx += 1

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        file = Path(self.processed_dir) / f"data_{idx}.pt"
        print(file)
        return torch.load(file, weights_only=False)

    def get_season_indices(self, season_name: str) -> List[int]:
        """Get indices for a specific season."""
        return [idx for idx in range(len(self)) if self.get(idx).season == season_name]

    def get_season_split(
        self, train_seasons: List[str], val_seasons: List[str], test_seasons: List[str]
    ):
        """Split by season names."""
        train_indices, val_indices, test_indices = [], [], []

        for idx in range(len(self)):
            season = self.get(idx).season
            if season in train_seasons:
                train_indices.append(idx)
            elif season in val_seasons:
                val_indices.append(idx)
            elif season in test_seasons:
                test_indices.append(idx)

        return train_indices, val_indices, test_indices


class SequentialSoccerDataset(SoccerDataset):
    def _process_match(
        self, events: pd.DataFrame, home_team: str, away_team: str
    ) -> List[Data]:
        return build_team_graphs_with_goals(
            events, home_team, away_team, self.time_interval, cumulative=False
        )

    @property
    def processed_dir(self) -> str:
        return Path(self.root).joinpath("processed_sequential").as_posix()


class CumulativeSoccerDataset(SoccerDataset):
    def _process_match(
        self, events: pd.DataFrame, home_team: str, away_team: str
    ) -> List[Data]:
        return build_team_graphs_with_goals(
            events, home_team, away_team, self.time_interval, cumulative=True
        )

    @property
    def processed_dir(self) -> str:
        return Path(self.root).joinpath("processed_cumulative").as_posix()


def main():
    # Test the improved version
    dataset = CumulativeSoccerDataset(
        root="../data", starting_year=2015, ending_year=2024, time_interval=30
    )
    print(f"Dataset length: {len(dataset)}")

    # Check the improved data structure
    if len(dataset) > 0:
        for i in range(min(5, len(dataset))):
            data = dataset[i]
            print(f"\nData point {i}:")
            print(f"Match: {data.home_team} vs {data.away_team}")
            print(f"Season: {data.season}, Time: {data.start_minute}-{data.end_minute}")
            print(
                f"Home graph: {data.home_x.shape[0]} players, {data.home_edge_index.shape[1]} passes"
            )
            print(
                f"Away graph: {data.away_x.shape[0]} players, {data.away_edge_index.shape[1]} passes"
            )
            print(f"Current goals: {data.current_home_goals}-{data.current_away_goals}")
            print(f"Final goals: {data.final_home_goals}-{data.final_away_goals}")
            print(f"Final result: {data.final_result}")
            print(f"All attributes: {list(data.keys())}")


if __name__ == "__main__":
    main()
