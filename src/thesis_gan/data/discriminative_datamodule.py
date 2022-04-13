import logging
from typing import List, Optional, Sequence, Union

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset

pylogger = logging.getLogger(__name__)


class MyDataModule(pl.LightningDataModule):
    def __init__(
        self,
        datasets: DictConfig,
        num_workers: DictConfig,
        batch_size: DictConfig,
        gpus: Optional[Union[List[int], str, int]],
    ):
        super().__init__()
        self.datasets = datasets
        self.num_workers = num_workers
        self.batch_size = batch_size
        # https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#gpus
        self.pin_memory: bool = gpus is not None and str(gpus) != "0"
        self.train_dataset: Optional[Dataset] = None
        self.val_datasets: Optional[Sequence[Dataset]] = None
        self.test_datasets: Optional[Sequence[Dataset]] = None

    def prepare_data(self) -> None:
        # download only
        pass

    def setup(self, stage: Optional[str] = None):
        # Here you should instantiate your datasets, you may also split the train into train and validation if needed.
        if (stage is None or stage == "fit") and (self.train_dataset is None and self.val_datasets is None):
            # example
            self.train_dataset = hydra.utils.instantiate(self.datasets.train, split="train")

            self.val_datasets = [hydra.utils.instantiate(dataset_cfg, split="val") for dataset_cfg in self.datasets.val]

        if stage is None or stage == "test":
            self.test_datasets = [
                hydra.utils.instantiate(dataset_cfg, split="test") for dataset_cfg in self.datasets.test
            ]

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.batch_size.train,
            num_workers=self.num_workers.train,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> Sequence[DataLoader]:
        return [
            DataLoader(
                dataset,
                shuffle=False,
                batch_size=self.batch_size.val,
                num_workers=self.num_workers.val,
                pin_memory=self.pin_memory,
            )
            for dataset in self.val_datasets
        ]

    def test_dataloader(self) -> Sequence[DataLoader]:
        return [
            DataLoader(
                dataset,
                shuffle=False,
                batch_size=self.batch_size.test,
                num_workers=self.num_workers.test,
                pin_memory=self.pin_memory,
            )
            for dataset in self.test_datasets
        ]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(" f"{self.datasets=}, " f"{self.num_workers=}, " f"{self.batch_size=})"
