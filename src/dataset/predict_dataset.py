from typing import Dict

import numpy as np
import torch

from dataset.dataset import StockDataset


class StockDatasetPredict(StockDataset):
    
    def __len__(self) -> int:
        return ((len(self.data) - (self.encoder_length + self.generation_length)) // self.stride) + 1

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        x_slice = slice(self.stride * index, self.stride * index + self.encoder_length)
        y_slice = slice(
            self.stride * index + self.encoder_length,
            self.stride * index + self.encoder_length + self.generation_length,
        )
        x = torch.as_tensor(self.data[x_slice].T, dtype=torch.float)
        y_price = torch.as_tensor(self.prices[y_slice].T, dtype=torch.float)
        y_volume = torch.as_tensor(self.volumes[y_slice].T, dtype=torch.float)
        t = torch.as_tensor(np.concat((self.t[x_slice], self.t[y_slice])), dtype=torch.float)
        return_dict = dict(x=x, y_price=y_price, y_volume=y_volume, t=t)
        return return_dict
