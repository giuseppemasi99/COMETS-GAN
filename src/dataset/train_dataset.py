from typing import Dict

import torch

from dataset.dataset import StockDataset


class StockDatasetTrain(StockDataset):

    def __len__(self) -> int:
        return ((len(self.data) - (self.encoder_length + self.decoder_length)) // self.stride) + 1

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        x_slice = slice(self.stride * index, self.stride * index + self.encoder_length)
        y_slice = slice(
            self.stride * index + self.encoder_length,
            self.stride * index + self.encoder_length + self.decoder_length,
        )
        x = torch.as_tensor(self.data[x_slice].T, dtype=torch.float)
        y = torch.as_tensor(self.data[y_slice].T, dtype=torch.float)

        t_past = torch.as_tensor(self.t[x_slice], dtype=torch.float)
        t_fut = torch.as_tensor(self.t[y_slice], dtype=torch.float)
        
        return_dict = dict(x=x, y=y, t_past=t_past, t_fut=t_fut)
        return return_dict