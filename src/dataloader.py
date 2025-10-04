import pickle
from pathlib import Path

import torch
from torch_geometric.data import Dataset


class SoccerDataset(Dataset):
    @property
    def raw_file_names(self) -> list[str]:
        return [f"epl_{year}.pkl" for year in range(2015, 2024)]

    def process(self) -> None:
        idx = 0
        for raw_path in self.raw_paths:
            with open(raw_path, "rb") as f:
                raw_data = pickle.load(f)

                if self.pre_filter is not None and not self.pre_filter(raw_data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(raw_data)

                torch.save(data, Path(self.processed_dir).joinpath(f"data_{idx}.pt"))
                idx += 1

    def len(self) -> int:
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(Path(self.processed_dir).joinpath(f"data_{idx}.pt"))
        return data
