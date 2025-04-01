from hydra.utils import instantiate
from lightning import LightningDataModule
from omegaconf.dictconfig import DictConfig
from torch.utils.data import DataLoader

from data.dataset_noise import DatasetNoise
from data.dataset_stocks import DatasetStocks
from data.preprocessing import Pipeline


class MyDataModule(LightningDataModule):

    def __init__(
        self, dataset_train: DictConfig, dataset_val: DictConfig, dataset_predict: DictConfig, pipeline: DictConfig,
        batch_size: int, num_workers: int, pin_memory: bool
    ) -> None:
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.pipeline: Pipeline = instantiate(pipeline)

        self.dataset_train: DatasetStocks = instantiate(dataset_train, pipeline=self.pipeline)
        self.dataset_val: DatasetStocks = instantiate(dataset_val, pipeline=self.pipeline)
        self.dataset_predict: DatasetNoise = instantiate(dataset_predict)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset_train, batch_size=self.batch_size, shuffle=True, 
            num_workers=self.num_workers, pin_memory=self.pin_memory, drop_last=True
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset_val, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=self.pin_memory, drop_last=True
        )

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset_predict, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=self.pin_memory, drop_last=True
        )
