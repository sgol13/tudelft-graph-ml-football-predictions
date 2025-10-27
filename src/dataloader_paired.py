import pickle
from collections import Counter, defaultdict
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Batch, Data, Dataset, HeteroData
from tqdm import tqdm


def count_passes(
    team_events: pd.DataFrame,
) -> tuple[Counter[tuple[int, int]], dict[str, tuple[float, float, float]], list[str]]:
    """Count successful passes within a team over given events."""
    if team_events.empty:
        return Counter(), {}, []

    team_players = team_events["player"].dropna().unique().astype(str)
    player_to_idx = {p: i for i, p in enumerate(team_players)}
    players_pass_positions_x = {player: [] for player in team_players}
    players_pass_positions_y = {player: [] for player in team_players}

    rows = team_events[
        ["type", "outcome_type", "player", "x", "y", "end_x", "end_y"]
    ].to_numpy()
    passes = []
    pass_from = None
    receiving_x = None
    receiving_y = None

    for event_type, outcome_type, player, x, y, end_x, end_y in rows:
        if event_type == "Pass" and outcome_type == "Successful":
            if player not in player_to_idx:
                continue
            if pass_from is not None and pass_from in player_to_idx:
                passes.append((player_to_idx[pass_from], player_to_idx[player]))
                if x is not None and y is not None:
                    players_pass_positions_x[pass_from].append(x)
                    players_pass_positions_y[pass_from].append(y)

                if receiving_x is not None and receiving_y is not None:
                    players_pass_positions_x[player].append(receiving_x)
                    players_pass_positions_y[player].append(receiving_y)

        pass_from = player
        receiving_x = end_x
        receiving_y = end_y

    pass_counts = Counter(passes)
    players_positions = {}
    for player in team_players:
        x_positions = players_pass_positions_x[player]
        y_positions = players_pass_positions_y[player]

        if x_positions and y_positions:
            mean_x = np.mean(x_positions)
            mean_y = np.mean(y_positions)
            is_valid = 1.0  # Valid position
        else:
            mean_x, mean_y = 50.0, 50.0  # Midfield
            is_valid = 0.0  # Invalid/imputed position

        players_positions[player] = (mean_x, mean_y, is_valid)

    return pass_counts, players_positions, team_players.tolist()


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

    # Create one-hot encoding
    if home_goals > away_goals:
        result_tensor = torch.tensor([1, 0, 0], dtype=torch.float)  # Home Win
    elif home_goals == away_goals:
        result_tensor = torch.tensor([0, 1, 0], dtype=torch.float)  # Draw
    else:
        result_tensor = torch.tensor([0, 0, 1], dtype=torch.float)  # Away Win

    return home_goals, away_goals, result_tensor


def build_graph(
    pass_counts: Counter, player_positions: dict, team_players: list
) -> Data:
    """Build graph - handles empty pass cases gracefully."""
    if not pass_counts or len(team_players) == 0:
        # Handle empty case
        x = torch.zeros((max(1, len(team_players)), 4), dtype=torch.float)
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_weight = torch.zeros(0, dtype=torch.float)
    else:
        # Build edges
        edge_index = (
            torch.tensor(list(pass_counts.keys()), dtype=torch.long).t().contiguous()
        )
        edge_weight = torch.tensor(list(pass_counts.values()), dtype=torch.float)

        # Enhanced node features: [player_id, position_x, position_y]
        x_list = []
        player_to_idx = {p: i for i, p in enumerate(team_players)}

        for player in team_players:
            player_id = player_to_idx[player]
            if player in player_positions:
                pos_x, pos_y, pos_valid = player_positions[player]
                x_list.append([float(player_id), pos_x, pos_y, pos_valid])
            else:
                x_list.append([float(player_id), 50.0, 50.0, 0.0])

        x = torch.tensor(x_list, dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_weight=edge_weight)


def build_team_graphs_with_goals(
    events, home_team: str, away_team: str, time_interval=5, cumulative=False
) -> List[HeteroData]:
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
    cumulative_player_positions = defaultdict(lambda: defaultdict(list))

    # Iterate over time intervals
    for start_minute in range(0, int(max_minute), time_interval):
        end_minute = min(start_minute + time_interval, int(max_minute) + 1)
        events_in_interval = events[
            (events["minute"] >= start_minute)
            & (events["minute"] < end_minute)
            & (~events["type"].isin(["Start", "End", "FormationSet"]))
        ]

        # Count goals up to current interval
        current_home_goals, current_away_goals = count_goals(
            events, end_minute, home_team, away_team
        )

        # Process HOME team
        home_events = events_in_interval[events_in_interval["team"] == home_team]
        home_pass_counts, home_positions, home_players = count_passes(home_events)

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

            # Save positions up to current interval
            for player, position in home_positions.items():
                if player in cumulative_player_positions[home_team]:
                    cumulative_player_positions[home_team][player].append(position)
                else:
                    cumulative_player_positions[home_team][player] = [position]

            # Calculate MEAN of cumulative positions up to current interval
            current_cumulative_positions = {}
            for player, positions_list in cumulative_player_positions[
                home_team
            ].items():
                if positions_list:
                    valid_positions = [
                        pos for pos in positions_list if pos[2] == 1.0
                    ]  # Only use valid ones
                    if valid_positions:
                        mean_x = np.mean([pos[0] for pos in valid_positions])
                        mean_y = np.mean([pos[1] for pos in valid_positions])
                        is_valid = 1.0
                    else:
                        mean_x, mean_y = 50.0, 50.0
                        is_valid = 0.0
                else:
                    mean_x, mean_y = 50.0, 50.0
                    is_valid = 0.0

                current_cumulative_positions[player] = (mean_x, mean_y, is_valid)

            home_positions = current_cumulative_positions

        home_graph = build_graph(home_pass_counts, home_positions, home_players)

        # Process AWAY team
        away_events = events_in_interval[events_in_interval["team"] == away_team]
        away_pass_counts, away_positions, away_players = count_passes(away_events)

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

            # Save positions up to current interval
            for player, position in away_positions.items():
                # position is (x, y, is_valid)
                if player in cumulative_player_positions[away_team]:
                    cumulative_player_positions[away_team][player].append(position)
                else:
                    cumulative_player_positions[away_team][player] = [position]

            # Calculate MEAN of cumulative positions up to current interval
            current_cumulative_positions = {}
            for player, positions_list in cumulative_player_positions[
                away_team
            ].items():
                if positions_list:
                    # Only average if we have valid positions
                    valid_positions = [
                        pos for pos in positions_list if pos[2] == 1.0
                    ]  # Only use valid ones
                    if valid_positions:
                        mean_x = np.mean([pos[0] for pos in valid_positions])
                        mean_y = np.mean([pos[1] for pos in valid_positions])
                        is_valid = 1.0
                    else:
                        mean_x, mean_y = 50.0, 50.0
                        is_valid = 0.0
                else:
                    mean_x, mean_y = 50.0, 50.0
                    is_valid = 0.0

                current_cumulative_positions[player] = (
                    mean_x,
                    mean_y,
                    is_valid,
                )

            away_positions = current_cumulative_positions

        away_graph = build_graph(away_pass_counts, away_positions, away_players)

        # Create HeteroData with separate node types for home/away
        hetero_data = HeteroData()

        # Home team graph
        hetero_data["home"].x = home_graph.x
        hetero_data["home", "passes_to", "home"].edge_index = home_graph.edge_index
        hetero_data["home", "passes_to", "home"].edge_weight = home_graph.edge_weight

        # Away team graph
        hetero_data["away"].x = away_graph.x
        hetero_data["away", "passes_to", "away"].edge_index = away_graph.edge_index
        hetero_data["away", "passes_to", "away"].edge_weight = away_graph.edge_weight

        # Metadata (stored as attributes on the HeteroData object)
        hetero_data.start_minute = torch.tensor(start_minute, dtype=torch.long)
        hetero_data.end_minute = torch.tensor(end_minute, dtype=torch.long)
        hetero_data.y = final_result_onehot  # Keep as one-hot for now
        hetero_data.current_home_goals = torch.tensor(
            current_home_goals, dtype=torch.long
        )
        hetero_data.current_away_goals = torch.tensor(
            current_away_goals, dtype=torch.long
        )
        hetero_data.final_home_goals = torch.tensor(final_home_goals, dtype=torch.long)
        hetero_data.final_away_goals = torch.tensor(final_away_goals, dtype=torch.long)
        hetero_data.home_team = home_team
        hetero_data.away_team = away_team

        paired_graphs.append(hetero_data)

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
    ) -> List[HeteroData]:
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

                paired_graphs = self._process_match(events, home_team, away_team)

                for hetero_data in paired_graphs:
                    # Add metadata as attributes directly on HeteroData
                    hetero_data.season = season_name
                    hetero_data.match_id = match_id
                    hetero_data.data_id = idx

                    if self.pre_filter is not None and not self.pre_filter(hetero_data):
                        continue
                    if self.pre_transform is not None:
                        hetero_data = self.pre_transform(hetero_data)

                    torch.save(hetero_data, Path(self.processed_dir) / f"data_{idx}.pt")
                    idx += 1

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        file = Path(self.processed_dir) / f"data_{idx}.pt"
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
    ) -> List[HeteroData]:
        return build_team_graphs_with_goals(
            events, home_team, away_team, self.time_interval, cumulative=False
        )

    @property
    def processed_dir(self) -> str:
        dir_name = f"processed_sequential_{self.starting_year}-{self.ending_year}-{self.time_interval}"
        return Path(self.root).joinpath(dir_name).as_posix()


class CumulativeSoccerDataset(SoccerDataset):
    def _process_match(
        self, events: pd.DataFrame, home_team: str, away_team: str
    ) -> List[HeteroData]:
        return build_team_graphs_with_goals(
            events, home_team, away_team, self.time_interval, cumulative=True
        )

    @property
    def processed_dir(self) -> str:
        dir_name = f"processed_cumulative_{self.starting_year}-{self.ending_year}-{self.time_interval}"
        return Path(self.root).joinpath(dir_name).as_posix()


class GroupedSoccerDataset(SequentialSoccerDataset):
    """
    Groups all timeframes per match together into one saved object.
    Each entry in the dataset corresponds to a match.
    """

    def process(self):
        idx = 0
        for raw_path in tqdm(self.raw_paths, desc="Processing seasons"):
            season_name = Path(raw_path).stem

            with open(raw_path, "rb") as f:
                season_data = pickle.load(f)

            for match in tqdm(
                season_data, desc=f"Matches from {season_name}", leave=False
            ):
                events = match["events"]
                home_team = match["home_team"]
                away_team = match["away_team"]
                match_id = match.get("game_id", f"{season_name}_{idx}")

                timeframes = self._process_match(events, home_team, away_team)

                processed_timeframes = []
                for hetero_data in timeframes:
                    hetero_data.season = season_name
                    hetero_data.match_id = match_id
                    hetero_data.data_id = idx

                    if self.pre_filter is not None and not self.pre_filter(hetero_data):
                        continue
                    if self.pre_transform is not None:
                        hetero_data = self.pre_transform(hetero_data)
                    processed_timeframes.append(hetero_data)

                # Save the whole list together
                torch.save(
                    processed_timeframes, Path(self.processed_dir) / f"match_{idx}.pt"
                )
                idx += 1

    @property
    def processed_dir(self) -> str:
        dir_name = f"processed_grouped_{self.starting_year}-{self.ending_year}-{self.time_interval}"
        return Path(self.root).joinpath(dir_name).as_posix()

    @property
    def processed_file_names(self):
        processed = list(Path(self.processed_dir).glob("match_*.pt"))
        return [p.name for p in processed]

    def get(self, idx) -> HeteroData:
        """Returns all timeframes for a single match as a batched graph."""
        file = Path(self.processed_dir) / f"match_{idx}.pt"
        timeframes: list[HeteroData] = torch.load(file, weights_only=False)

        # Batch them into one HeteroData object
        batched = Batch.from_data_list(timeframes)
        batched.match_id = timeframes[0].match_id
        batched.season = timeframes[0].season
        return batched


def main():
    # Test the improved version
    dataset = GroupedSoccerDataset(
        root="data", starting_year=2015, ending_year=2016, time_interval=30
    )
    print(f"Dataset length: {len(dataset)}")

    # Check the improved data structure
    if len(dataset) > 0:
        for i in range(min(5, len(dataset))):
            data = dataset[i]
            print(data)


if __name__ == "__main__":
    main()
