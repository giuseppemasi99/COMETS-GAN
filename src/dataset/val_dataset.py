from typing import Dict

import torch

from dataset.dataset import StockDataset


class StockDatasetVal(StockDataset):

    def __len__(self) -> int:
        return 1

    def __getitem__(self, _: int) -> Dict[str, torch.Tensor]:
        x = torch.as_tensor(self.data.T, dtype=torch.float)
        t = torch.as_tensor(self.t, dtype=torch.float)
        return_dict = dict(x=x, t=t)
        return return_dict
