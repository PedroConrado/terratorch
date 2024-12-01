from typing import Any

import albumentations as A
from torch import Tensor
from torch.utils.data import DataLoader
from torchgeo.datamodules import NonGeoDataModule

from terratorch.datamodules.generic_pixel_wise_data_module import Normalize
from terratorch.datamodules.utils import wrap_in_compose_is_list
from terratorch.datasets import MultiTemporalCropClassification


MEANS = {
    "BLUE": 494.905781,
    "GREEN": 815.239594,
    "RED": 924.335066,
    "NIR_NARROW": 2968.881459,
    "SWIR_1": 2634.621962,
    "SWIR_2": 1739.579917,
}

STDS = {
    "BLUE": 284.925432,
    "GREEN": 357.84876,
    "RED": 575.566823,
    "NIR_NARROW": 896.601013,
    "SWIR_1": 951.900334,
    "SWIR_2": 921.407808,
}


class MultiTemporalCropClassificationDataModule(NonGeoDataModule):
    """NonGeo datamodule implementation for multi-temporal crop classification."""

    def __init__(
        self,
        data_root: str,
        batch_size: int = 4,
        num_workers: int = 0,
        train_transform: A.Compose | None | list[A.BasicTransform] = None,
        val_transform: A.Compose | None | list[A.BasicTransform] = None,
        test_transform: A.Compose | None | list[A.BasicTransform] = None,
        drop_last: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(MultiTemporalCropClassification, batch_size, num_workers, **kwargs)
        self.data_root = data_root

        bands = kwargs.get("bands", MultiTemporalCropClassification.all_band_names)
        means = [MEANS[b] for b in bands]
        stds = [STDS[b] for b in bands]

        self.train_transform = wrap_in_compose_is_list(train_transform)
        self.val_transform = wrap_in_compose_is_list(val_transform)
        self.test_transform = wrap_in_compose_is_list(test_transform)
        self.aug = Normalize(means, stds)
        self.drop_last = drop_last

    def setup(self, stage: str) -> None:
        if stage in ["fit"]:
            self.train_dataset = self.dataset_class(
                split="train",
                data_root=self.data_root,
                transform=self.train_transform,
                **self.kwargs,
            )
        if stage in ["fit", "validate"]:
            self.val_dataset = self.dataset_class(
                split="val",
                data_root=self.data_root,
                transform=self.val_transform,
                **self.kwargs,
            )
        if stage in ["test"]:
            self.test_dataset = self.dataset_class(
                split="test",
                data_root=self.data_root,
                transform=self.test_transform,
                **self.kwargs,
            )

    def _dataloader_factory(self, split: str) -> DataLoader[dict[str, Tensor]]:
        """Implement one or more PyTorch DataLoaders.

        Args:
            split: Either 'train', 'val', 'test', or 'predict'.

        Returns:
            A collection of data loaders specifying samples.

        Raises:
            MisconfigurationException: If :meth:`setup` does not define a
                dataset or sampler, or if the dataset or sampler has length 0.
        """
        dataset = self._valid_attribute(f"{split}_dataset", "dataset")
        batch_size = self._valid_attribute(f"{split}_batch_size", "batch_size")
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=split == "train",
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            drop_last=split == "train" and self.drop_last,
        )
