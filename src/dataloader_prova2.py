import pickle
from collections import Counter, defaultdict
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, Dataset
from tqdm import tqdm


def count_passes(team_events: pd.DataFrame) -> tuple[Counter[tuple[int, int]], np.ndarray]:
    """Count successful passes within a team over given events."""
    if team_events.empty:
        return Counter(), np.array([]).astype(str)

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
    return pass_counts, team_players


def build_graph(pass_counts: Counter, team_players: list[str]) -> Data:
    """Build graph - handles empty pass cases gracefully."""
    if not pass_counts:
        # Return empty graph instead of crashing
        x = torch.zeros((max(1, len(team_players)), 1), dtype=torch.float)
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_weight = torch.zeros(0, dtype=torch.float)
    else:
        edge_index = torch.tensor(list(pass_counts.keys()), dtype=torch.long).t().contiguous()
        edge_weight = torch.tensor(list(pass_counts.values()), dtype=torch.float)
        x = torch.arange(len(team_players), dtype=torch.float).unsqueeze(1)

    return Data(x=x, edge_index=edge_index, edge_weight=edge_weight)


def build_paired_graphs(
    events: pd.DataFrame, 
    home_team: str,
    away_team: str,
    time_interval: int = 5, 
    cumulative: bool = False
) -> List[Data]:
    """Build PAIRED graphs for home and away teams for each time interval."""
    # Check if required data exists
    if events.empty or 'minute' not in events.columns:
        return []
    
    max_minute = events["minute"].max()
    
    paired_graphs = []
    cumulative_pass_counts = defaultdict(Counter)
    cumulative_team_players = defaultdict(set)

    for start_minute in range(0, max_minute, time_interval):
        end_minute = min(start_minute + time_interval, max_minute + 1)
        events_in_interval = events[
            (events["minute"] >= start_minute) &
            (events["minute"] < end_minute) &
            (~events["type"].isin(["Start", "End", "FormationSet"]))
        ]

        # Process HOME team
        home_events = events_in_interval[events_in_interval["team"] == home_team]
        home_pass_counts, home_players = count_passes(home_events)

        if cumulative and home_pass_counts:
            for (i, j), count in home_pass_counts.items():
                p_from, p_to = home_players[i], home_players[j]
                cumulative_pass_counts[home_team][(p_from, p_to)] += count
            cumulative_team_players[home_team].update(home_players)
            home_players = sorted(cumulative_team_players[home_team])
            player_to_idx = {p: i for i, p in enumerate(home_players)}
            home_pass_counts = Counter({
                (player_to_idx[i], player_to_idx[j]): count
                for (i, j), count in cumulative_pass_counts[home_team].items()
                if i in player_to_idx and j in player_to_idx
            })

        home_graph = build_graph(home_pass_counts, home_players)

        # Process AWAY team
        away_events = events_in_interval[events_in_interval["team"] == away_team]
        away_pass_counts, away_players = count_passes(away_events)

        if cumulative and away_pass_counts:
            for (i, j), count in away_pass_counts.items():
                p_from, p_to = away_players[i], away_players[j]
                cumulative_pass_counts[away_team][(p_from, p_to)] += count
            cumulative_team_players[away_team].update(away_players)
            away_players = sorted(cumulative_team_players[away_team])
            player_to_idx = {p: i for i, p in enumerate(away_players)}
            away_pass_counts = Counter({
                (player_to_idx[i], player_to_idx[j]): count
                for (i, j), count in cumulative_pass_counts[away_team].items()
                if i in player_to_idx and j in player_to_idx
            })

        away_graph = build_graph(away_pass_counts, away_players)

        # Create PAIRED data point with both teams
        paired_data = Data(
            # Home team graph data
            home_x=home_graph.x,
            home_edge_index=home_graph.edge_index,
            home_edge_weight=home_graph.edge_weight,
            
            # Away team graph data
            away_x=away_graph.x,
            away_edge_index=away_graph.edge_index,
            away_edge_weight=away_graph.edge_weight,
            
            # Time information
            start_minute=torch.tensor(start_minute, dtype=torch.long),
            end_minute=torch.tensor(end_minute, dtype=torch.long),
            
            # Store player lists as attributes (not tensors)
            home_players=home_players,
            away_players=away_players,
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

    def _process_match(self, events: pd.DataFrame, home_team: str, away_team: str, season: str) -> List[Data]:
        raise NotImplementedError()

    def process(self):
        idx = 0
        
        for raw_path in tqdm(self.raw_paths, desc="Processing seasons"):
            season_name = Path(raw_path).stem  # e.g., "epl_2015"
            
            with open(raw_path, "rb") as f:
                season_data = pickle.load(f)

            for match in tqdm(season_data, desc=f"Matches from {season_name}", leave=False):
                events = match["events"]
                home_team = match["home_team"]
                away_team = match["away_team"]
                match_id = match.get("game_id", f"{season_name}_{idx}")
                
                # Get PAIRED graphs for this match
                paired_graphs = self._process_match(events, home_team, away_team, season_name)
                
                for paired_data in paired_graphs:
                    # Store rich context information
                    paired_data.season = season_name
                    paired_data.match_id = match_id
                    paired_data.home_team_name = home_team
                    paired_data.away_team_name = away_team
                    paired_data.data_id = idx
                    
                    # You can also add match outcomes if available
                    # paired_data.result = match.get("result", None)  # "H", "D", "A"
                    
                    if self.pre_filter is not None and not self.pre_filter(paired_data):
                        continue
                    if self.pre_transform is not None:
                        paired_data = self.pre_transform(paired_data)
                    
                    torch.save(paired_data, Path(self.processed_dir) / f"data_{idx}.pt")
                    idx += 1

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        return torch.load(Path(self.processed_dir) / f"data_{idx}.pt")
    
    def get_season_indices(self, season_name: str) -> List[int]:
        """Get indices for a specific season."""
        return [idx for idx in range(len(self)) if self.get(idx).season == season_name]
    
    def get_season_split(self, train_seasons: List[str], val_seasons: List[str], test_seasons: List[str]):
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
    def _process_match(self, events: pd.DataFrame, home_team: str, away_team: str, season: str) -> List[Data]:
        return build_paired_graphs(
            events, home_team, away_team, self.time_interval, cumulative=False
        )


class CumulativeSoccerDataset(SoccerDataset):
    def _process_match(self, events: pd.DataFrame, home_team: str, away_team: str, season: str) -> List[Data]:
        return build_paired_graphs(
            events, home_team, away_team, self.time_interval, cumulative=True
        )


# Test the improved version
dataset = CumulativeSoccerDataset(root="../data", starting_year=2015, ending_year=2016, time_interval=30)
print(f"Dataset length: {len(dataset)}")

# Check the improved data structure
if len(dataset) > 0:
    for i in range(min(5, len(dataset))):
        data = dataset[i]
        print(f"\nData point {i}:")
        print(f"Match: {data.home_team_name} vs {data.away_team_name}")
        print(f"Season: {data.season}, Time: {data.start_minute}-{data.end_minute}")
        print(f"Home graph: {data.home_x.shape[0]} players, {data.home_edge_index.shape[1]} passes")
        print(f"Away graph: {data.away_x.shape[0]} players, {data.away_edge_index.shape[1]} passes")
        print(f"All attributes: {list(data.keys())}")