import pickle
from pathlib import Path

import torch
from pandas import DataFrame
from torch_geometric.data import Data, Dataset


class SoccerDataset(Dataset):
    def __init__(self, root, pre_filter=None, pre_transform=None, time_interval=5):
        super().__init__(root=root, pre_filter=pre_filter, pre_transform=pre_transform)
        self.time_interval = time_interval
        self.starting_year = 2015
        self.ending_year = 2015
        self.matches_per_year = 380

    @property
    def raw_file_names(self):
        return [
            f"epl_{year}.pkl"
            for year in range(self.starting_year, self.ending_year + 1)
        ]

    @property
    def processed_file_names(self):
        num_years = self.ending_year - self.starting_year + 1
        num_matches = self.matches_per_year * num_years
        return [f"data_{match}.pt" for match in range(num_matches)]

    def _process_match(self, events: DataFrame) -> Data:
        # See progress in test.ipynb, that functionality should be here.
        raise NotImplementedError()

    def process(self):
        idx = 0
        for raw_path in self.raw_paths:
            with open(raw_path, "rb") as f:
                raw_data = pickle.load(f)
                for raw_match in raw_data:
                    events = raw_match["events"]

                    data = self._process_match(events)

                    if self.pre_filter is not None and not self.pre_filter(raw_data):
                        continue

                    if self.pre_transform is not None:
                        data = self.pre_transform(raw_data)

                    torch.save(
                        data, Path(self.processed_dir).joinpath(f"data_{idx}.pt")
                    )
                    idx += 1

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(Path(self.processed_dir).joinpath(f"data_{idx}.pt"))
        return data
