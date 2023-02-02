import logging
import pickle
from pathlib import Path
from typing import List, Optional, Sequence, Union

import hydra
import omegaconf
import pytorch_lightning as pl
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset

from nn_core.common import PROJECT_ROOT

from thesis_gan.data.pipeline import Pipeline

pylogger = logging.getLogger(__name__)


class MetaData:
    def __init__(self, data_pipeline_price: Pipeline, data_pipeline_volume: Pipeline):
        """The data information the Lightning Module will be provided with.

        This is a "bridge" between the Lightning DataModule and the Lightning Module.
        There is no constraint on the class name nor in the stored information, as long as it exposes the
        `save` and `load` methods.

        The Lightning Module will receive an instance of MetaData when instantiated,
        both in the train loop or when restored from a checkpoint.

        This decoupling allows the architecture to be parametric (e.g. in the number of classes) and
        DataModule/Trainer independent (useful in prediction scenarios).
        MetaData should contain all the information needed at test time, derived from its train dataset.

        Examples are the class names in a classification task or the vocabulary in NLP tasks.
        MetaData exposes `save` and `load`. Those are two user-defined methods that specify
        how to serialize and de-serialize the information contained in its attributes.
        This is needed for the checkpointing restore to work properly.

        Args:
            data_pipeline_price: pipeline used to preprocess the prices and to inverse transform the outputs of the model.
            data_pipeline_volume: pipeline used to preprocess the volumes and to inverse transform the outputs of the model.
        """
        self.data_pipeline_price = data_pipeline_price
        self.data_pipeline_volume = data_pipeline_volume

    def save(self, dst_path: Path) -> None:
        """Serialize the MetaData attributes into the zipped checkpoint in dst_path.

        Args:
            dst_path: the root folder of the metadata inside the zipped checkpoint
        """
        pylogger.debug(f"Saving MetaData to '{dst_path}'")
        with open(dst_path / "data_pipeline.pickle", "wb") as f:
            pickle.dump(
                {
                    "data_pipeline_price": self.data_pipeline_price,
                    "data_pipeline_volume": self.data_pipeline_volume,
                },
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    @staticmethod
    def load(src_path: Path) -> "MetaData":
        """Deserialize the MetaData from the information contained inside the zipped checkpoint in src_path.

        Args:
            src_path: the root folder of the metadata inside the zipped checkpoint

        Returns:
            an instance of MetaData containing the information in the checkpoint
        """
        pylogger.debug(f"Loading MetaData from '{src_path}'")

        with open(src_path / "data_pipeline.pickle", "rb") as f:
            data_pipelines = pickle.load(f)

        data_pipeline_price = data_pipelines["data_pipeline_price"]
        data_pipeline_volume = data_pipelines["data_pipeline_volume"]

        return MetaData(data_pipeline_price=data_pipeline_price, data_pipeline_volume=data_pipeline_volume)


class MyDataModule(pl.LightningDataModule):
    def __init__(
        self,
        datasets: DictConfig,
        data_pipeline_price: DictConfig,
        data_pipeline_volume: DictConfig,
        num_workers: DictConfig,
        batch_size: DictConfig,
        gpus: Optional[Union[List[int], str, int]],
        dataset_type: str,
    ) -> None:
        super().__init__()
        self.datasets = datasets
        self.data_pipeline_price = data_pipeline_price
        self.data_pipeline_volume = data_pipeline_volume
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.dataset_type = dataset_type
        # https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#gpus
        self.pin_memory: bool = gpus is not None and str(gpus) != "0"
        self.train_dataset: Optional[Dataset] = None
        self.val_datasets: Optional[Sequence[Dataset]] = None
        self.test_datasets: Optional[Sequence[Dataset]] = None

    @property
    def metadata(self) -> MetaData:
        """Data information to be fed to the Lightning Module as parameter.

        Examples are vocabularies, number of classes...

        Returns:
            metadata: everything the model should know about the data, wrapped in a MetaData object.
        """
        # Since MetaData depends on the training data, we need to ensure the setup method has been called.
        if self.train_dataset is None:
            self.setup(stage="fit")

        return MetaData(
            data_pipeline_price=self.train_dataset.data_pipeline_price,
            data_pipeline_volume=self.train_dataset.data_pipeline_volume,
        )

    def prepare_data(self) -> None:
        # download only
        pass

    def setup(self, stage: Optional[str] = None):

        data_pipeline_price = hydra.utils.instantiate(self.data_pipeline_price)
        data_pipeline_volume = hydra.utils.instantiate(self.data_pipeline_volume)

        if (stage is None or stage == "fit") and (self.train_dataset is None and self.val_datasets is None):
            self.train_dataset = hydra.utils.instantiate(
                self.datasets.train,
                data_pipeline_price=data_pipeline_price,
                data_pipeline_volume=data_pipeline_volume,
                split="train",
            )  # , path=PROJECT_ROOT / "data"
            # )
            self.val_datasets = [
                hydra.utils.instantiate(
                    dataset_cfg,
                    data_pipeline_price=data_pipeline_price,
                    data_pipeline_volume=data_pipeline_volume,
                    split="val",
                )  # , path=PROJECT_ROOT / "data"
                # )
                for dataset_cfg in self.datasets.val
            ]

        if stage is None or stage == "test":
            self.test_datasets = [
                hydra.utils.instantiate(
                    dataset_cfg,
                    split="test",
                    data_pipeline_price=data_pipeline_price,
                    data_pipeline_volume=data_pipeline_volume,
                )  # , path=PROJECT_ROOT / "data"
                # )
                for dataset_cfg in self.datasets.test
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


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default", version_base=None)
def main(cfg: omegaconf.DictConfig) -> None:
    """Debug main to quickly develop the DataModule.

    Args:
        cfg: the hydra configuration
    """
    # data_pipeline: Pipeline = hydra.utils.instantiate(cfg.model.data.data_pipeline)
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(cfg.data.module, _recursive_=False)
    datamodule.setup("fit")
    print(datamodule.train_dataset.data_pipeline_price)
    print(datamodule.train_dataset.data_pipeline_volume)
    # datamodule.metadata


if __name__ == "__main__":
    main()
