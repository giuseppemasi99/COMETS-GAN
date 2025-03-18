from hydra.utils import instantiate
from lightning import LightningDataModule
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from dataset.dataset import StockDataset
from dataset.pipeline import Pipeline


class MyDataModule(LightningDataModule):
    def __init__(
        self, dataset_train: DictConfig, dataset_val: DictConfig,
        pipeline_price: DictConfig, pipeline_volume: DictConfig,
        batch_size: int, num_workers: int, pin_memory: bool
    ) -> None:
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        
        self.pipeline_price: Pipeline = instantiate(pipeline_price)
        self.pipeline_volume: Pipeline = instantiate(pipeline_volume)

        self.dataset_train: StockDataset = instantiate(dataset_train, pipeline_price=self.pipeline_price, pipeline_volume=self.pipeline_volume)
        self.dataset_val: StockDataset = instantiate(dataset_val, pipeline_price=self.pipeline_price, pipeline_volume=self.pipeline_volume)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset_train, batch_size=self.batch_size, shuffle=True, 
            num_workers=self.num_workers, pin_memory=self.pin_memory, drop_last=True
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset_val, batch_size=self.batch_size, shuffle=True, 
            num_workers=self.num_workers, pin_memory=self.pin_memory
        )
