import pickle
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, Dataset, HeteroData
from tqdm import tqdm


@dataclass
class TemporalSequence:
    hetero_data_sequence: List[HeteroData]  # [t0, t1, t2, ...]
    y: torch.Tensor  # Single label for the entire sequence
    final_home_goals: torch.Tensor
    final_away_goals: torch.Tensor
    season: str
    match_id: str
    idx: int


def count_passes(
    team_events: pd.DataFrame,
) -> tuple[Counter[tuple[int, int]], dict[str, tuple[float, float, float]], list[str]]:
    """Count successful passes within a team over given events."""
    if team_events.empty:
        return Counter(), {}, []

    pass_events = team_events[
        (team_events["type"] == "Pass") & (team_events["outcome_type"] == "Successful")
    ]
    assert type(pass_events) is pd.DataFrame

    passing_players = set()
    rows = pass_events[["player", "x", "y", "end_x", "end_y"]].to_numpy()

    for player, x, y, end_x, end_y in rows:
        if player is not None and player is not np.nan and pd.notna(player):
            passing_players.add(str(player))

    if not passing_players:
        return Counter(), {}, []

    team_players = sorted(list(passing_players))
    player_to_idx = {p: i for i, p in enumerate(team_players)}

    players_pass_positions_x = {player: [] for player in team_players}
    players_pass_positions_y = {player: [] for player in team_players}

    # Procesar pases
    rows = team_events[
        ["type", "outcome_type", "player", "x", "y", "end_x", "end_y"]
    ].to_numpy()
    passes = []
    pass_from = None
    receiving_x = None
    receiving_y = None

    for event_type, outcome_type, player, x, y, end_x, end_y in rows:
        if event_type == "Pass" and outcome_type == "Successful":

            if (
                player is None
                or player is np.nan
                or pd.isna(player)
                or player not in player_to_idx
            ):
                continue

            if pass_from is not None and pass_from in player_to_idx:
                passes.append((player_to_idx[pass_from], player_to_idx[player]))

                if (
                    x is not None
                    and x is not np.nan
                    and pd.notna(x)
                    and y is not None
                    and y is not np.nan
                    and pd.notna(y)
                ):
                    players_pass_positions_x[pass_from].append(float(x))
                    players_pass_positions_y[pass_from].append(float(y))

                if (
                    receiving_x is not None
                    and receiving_x is not np.nan
                    and pd.notna(receiving_x)
                    and receiving_y is not None
                    and receiving_y is not np.nan
                    and pd.notna(receiving_y)
                ):
                    players_pass_positions_x[player].append(float(receiving_x))
                    players_pass_positions_y[player].append(float(receiving_y))

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
            is_valid = 1.0
        else:
            mean_x, mean_y = 50.0, 50.0
            is_valid = 0.0

        players_positions[player] = (mean_x, mean_y, is_valid)

    return pass_counts, players_positions, team_players


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
    """Build paired graphs with both home and away teams for each time interval.

    cumulative: if True, pass counts are accumulated across intervals,
                but node indices are recalculated per interval to avoid index inflation.
    """
    if events.empty or "minute" not in events.columns:
        return []

    # Check time
    events = events.copy()
    events = events[
        (events["minute"].notnull())
        & (events["minute"] >= 0)
        & (events["minute"] < 180)
    ]
    if events.empty:
        raise Exception(f"No valid events for {home_team} vs {away_team}")

    max_minute = int(events["minute"].max())
    max_minute = min(max_minute, 120)

    # Get final result with one-hot encoding
    final_home_goals, final_away_goals, final_result_onehot = get_final_result(
        events, home_team, away_team
    )

    paired_graphs = []

    # Cumulative storage
    cumulative_pass_counts = defaultdict(Counter)
    cumulative_team_players = defaultdict(set)
    cumulative_player_positions = defaultdict(lambda: defaultdict(list))

    for start_minute in range(0, int(max_minute) + 1, time_interval):
        end_minute = min(start_minute + time_interval, int(max_minute) + 1)
        events_in_interval = events[
            (events["minute"] >= start_minute)
            & (events["minute"] < end_minute)
            & (~events["type"].isin(["Start", "End", "FormationSet"]))
        ]

        # --- HOME TEAM ---
        home_events = events_in_interval[events_in_interval["team"] == home_team]
        home_pass_counts, home_positions, home_players = count_passes(home_events)

        if cumulative and home_pass_counts:
            # Add to cumulative
            for (i, j), count in home_pass_counts.items():
                p_from = home_players[i]
                p_to = home_players[j]
                cumulative_pass_counts[home_team][(p_from, p_to)] += count
            cumulative_team_players[home_team].update(home_players)
            home_players = sorted(list(cumulative_team_players[home_team]))
            player_to_idx = {p: i for i, p in enumerate(home_players)}
            # Remap cumulative edges to new indices
            home_pass_counts = Counter(
                {
                    (player_to_idx[i], player_to_idx[j]): c
                    for (i, j), c in cumulative_pass_counts[home_team].items()
                    if i in player_to_idx and j in player_to_idx
                }
            )
            # Update cumulative positions
            for player, pos in home_positions.items():
                cumulative_player_positions[home_team][player].append(pos)
            # Mean positions
            current_positions = {}
            for player, pos_list in cumulative_player_positions[home_team].items():
                valid_pos = [p for p in pos_list if p[2] == 1.0]
                if valid_pos:
                    mean_x = np.mean([p[0] for p in valid_pos])
                    mean_y = np.mean([p[1] for p in valid_pos])
                    is_valid = 1.0
                else:
                    mean_x, mean_y, is_valid = 50.0, 50.0, 0.0
                current_positions[player] = (mean_x, mean_y, is_valid)
            home_positions = current_positions

        home_graph = build_graph(home_pass_counts, home_positions, home_players)

        # --- AWAY TEAM ---
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
                    (player_to_idx[i], player_to_idx[j]): c
                    for (i, j), c in cumulative_pass_counts[away_team].items()
                    if i in player_to_idx and j in player_to_idx
                }
            )
            for player, pos in away_positions.items():
                cumulative_player_positions[away_team][player].append(pos)
            current_positions = {}
            for player, pos_list in cumulative_player_positions[away_team].items():
                valid_pos = [p for p in pos_list if p[2] == 1.0]
                if valid_pos:
                    mean_x = np.mean([p[0] for p in valid_pos])
                    mean_y = np.mean([p[1] for p in valid_pos])
                    is_valid = 1.0
                else:
                    mean_x, mean_y, is_valid = 50.0, 50.0, 0.0
                current_positions[player] = (mean_x, mean_y, is_valid)
            away_positions = current_positions

        away_graph = build_graph(away_pass_counts, away_positions, away_players)

        # --- Create HeteroData ---
        hetero_data = HeteroData()
        hetero_data["home"].x = home_graph.x
        hetero_data["home", "passes_to", "home"].edge_index = home_graph.edge_index
        hetero_data["home", "passes_to", "home"].edge_weight = home_graph.edge_weight

        hetero_data["away"].x = away_graph.x
        hetero_data["away", "passes_to", "away"].edge_index = away_graph.edge_index
        hetero_data["away", "passes_to", "away"].edge_weight = away_graph.edge_weight

        # Metadata
        hetero_data.start_minute = torch.tensor(start_minute, dtype=torch.long)
        hetero_data.end_minute = torch.tensor(end_minute, dtype=torch.long)
        hetero_data.y = final_result_onehot
        hetero_data.current_home_goals = torch.tensor(
            count_goals(events, end_minute, home_team, away_team)[0], dtype=torch.long
        )
        hetero_data.current_away_goals = torch.tensor(
            count_goals(events, end_minute, home_team, away_team)[1], dtype=torch.long
        )
        hetero_data.final_home_goals = torch.tensor(final_home_goals, dtype=torch.long)
        hetero_data.final_away_goals = torch.tensor(final_away_goals, dtype=torch.long)
        hetero_data.home_team = home_team
        hetero_data.away_team = away_team

        paired_graphs.append(hetero_data)

    return paired_graphs


def build_team_graphs_progressive(
    events, home_team: str, away_team: str, time_interval=5
) -> TemporalSequence:
    """Build paired graphs with both home and away teams for each time interval."""
    if events.empty or "minute" not in events.columns:
        raise Exception("Faulty data entry")

    # Check time
    events = events.copy()
    events = events[
        (events["minute"].notnull())
        & (events["minute"] >= 0)
        & (events["minute"] < 180)
    ]
    if events.empty:
        raise Exception(f"No valid events for {home_team} vs {away_team}")

    max_minute = int(events["minute"].max())
    max_minute = min(max_minute, 120)

    final_home_goals, final_away_goals, final_result_onehot = get_final_result(
        events, home_team, away_team
    )

    sequences = []

    # Iterate over time intervals
    for start_minute in range(0, int(max_minute), time_interval):
        end_minute = min(start_minute + time_interval, int(max_minute) + 1)
        events_in_interval = events[
            (events["minute"] >= start_minute)
            & (events["minute"] < end_minute)
            & (~events["type"].isin(["Start", "End", "FormationSet"]))
        ]

        current_home_goals, current_away_goals = count_goals(
            events, end_minute, home_team, away_team
        )

        # Build home graph
        home_events = events_in_interval[events_in_interval["team"] == home_team]
        home_pass_counts, home_positions, home_players = count_passes(home_events)
        home_graph = build_graph(home_pass_counts, home_positions, home_players)

        # Build away graph
        away_events = events_in_interval[events_in_interval["team"] == away_team]
        away_pass_counts, away_positions, away_players = count_passes(away_events)
        away_graph = build_graph(away_pass_counts, away_positions, away_players)

        # Build HeteroData
        hetero_data = HeteroData()
        hetero_data["home"].x = home_graph.x
        hetero_data["home", "passes_to", "home"].edge_index = home_graph.edge_index
        hetero_data["home", "passes_to", "home"].edge_weight = home_graph.edge_weight

        hetero_data["away"].x = away_graph.x
        hetero_data["away", "passes_to", "away"].edge_index = away_graph.edge_index
        hetero_data["away", "passes_to", "away"].edge_weight = away_graph.edge_weight

        # Add metadata
        hetero_data.start_minute = torch.tensor(start_minute, dtype=torch.long)
        hetero_data.end_minute = torch.tensor(end_minute, dtype=torch.long)
        hetero_data.current_home_goals = torch.tensor(
            current_home_goals, dtype=torch.long
        )
        hetero_data.current_away_goals = torch.tensor(
            current_away_goals, dtype=torch.long
        )
        hetero_data.home_team = home_team
        hetero_data.away_team = away_team

        sequences.append(hetero_data)

    # Create TemporalSequence with **just this interval**
    sequence = TemporalSequence(
        hetero_data_sequence=sequences,
        y=final_result_onehot,
        final_home_goals=torch.tensor(final_home_goals, dtype=torch.long),
        final_away_goals=torch.tensor(final_away_goals, dtype=torch.long),
        season="",
        match_id="",
        idx=0,
    )

    return sequence

def sanity_check_events(events: pd.DataFrame, match_id: str):
    try:
        # Defensive copy
        events = events.copy()


        # Check for missing or weird values
        bad_minutes = events[
            (events["minute"].isna())
            | (events["minute"] < 0)
            | (events["minute"] > 180)
        ]
        if not bad_minutes.empty:
            print(f"⚠️  Weird minutes in match {match_id}:")
            print(bad_minutes[["minute", "type", "team", "player"]].head(10))
    except:
        print("Problem intransically in the data")


def sanity_check_positions(events: pd.DataFrame, match_id: str):
    try:
        # Drop rows with no spatial data
        valid_positions = events.dropna(subset=["x", "y", "end_x", "end_y"])

        if valid_positions.empty:
            print(f"⚠️  No valid positions in match {match_id}")
            return

        # Check ranges
        for col in ["x", "y", "end_x", "end_y"]:
            invalid = valid_positions[(valid_positions[col] < 0) | (valid_positions[col] > 100)]
            if not invalid.empty:
                print(f"⚠️  {len(invalid)} invalid '{col}' entries in match {match_id}")
                print(invalid[[col, "minute", "type", "player"]].head(5))
    except:
        print("Problem intransically in the data")



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
    ) -> List[HeteroData] | TemporalSequence:
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

                # assert type(paired_graphs) is List[HeteroData]

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


class TemporalSoccerDataset(SoccerDataset):
    @property
    def processed_dir(self):
        dir_name = f"processed_temporal_{self.starting_year}-{self.ending_year}-{self.time_interval}"
        return Path(self.root).joinpath(dir_name).as_posix()

    @property
    def processed_file_names(self):
        processed = list(Path(self.processed_dir).glob("sequence_*.pt"))
        return [p.name for p in processed]

    def _process_match(
        self, events: pd.DataFrame, home_team: str, away_team: str
    ) -> TemporalSequence:
        return build_team_graphs_progressive(
            events, home_team, away_team, self.time_interval
        )

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

                # sanity_check_events(events, match_id)
                # sanity_check_positions(events, match_id)

                # Get entire temporal sequence for this match
                try:
                    sequence = self._process_match(events, home_team, away_team)
                except Exception:
                    continue

                sequence.season = season_name
                sequence.match_id = match_id
                sequence.idx = idx

                if self.pre_filter and not self.pre_filter(sequence):
                    continue
                if self.pre_transform:
                    sequence = self.pre_transform(sequence)

                torch.save(sequence, Path(self.processed_dir) / f"sequence_{idx}.pt")
                idx += 1

    def get(self, idx) -> TemporalSequence:
        file = Path(self.processed_dir) / f"sequence_{idx}.pt"
        return torch.load(file, weights_only=False)


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


def main():
    dataset = TemporalSoccerDataset(
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
