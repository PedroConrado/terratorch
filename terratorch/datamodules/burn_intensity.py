from collections.abc import Sequence
from typing import Any

import albumentations as A
from torchgeo.datamodules import NonGeoDataModule

from terratorch.datamodules.generic_pixel_wise_data_module import Normalize
from terratorch.datamodules.utils import wrap_in_compose_is_list
from terratorch.datasets import BurnIntensityNonGeo

MEANS = {
    "BLUE": 1027.0479,
    "GREEN": 1230.4219,
    "RED": 1270.1404,
    "NIR": 1895.1371,
    "SWIR_1": 1366.3008,
    "SWIR_2": 2467.9395,
}

STDS = {
    "BLUE": 1735.9897,
    "GREEN": 1725.6100,
    "RED": 1687.5278,
    "NIR": 1252.5571,
    "SWIR_1": 1122.0760,
    "SWIR_2": 1522.4668,
}

class BurnIntensityNonGeoDataModule(NonGeoDataModule):
    """NonGeo datamodule implementation for BurnIntensity."""

    def __init__(
        self,
        data_root: str,
        batch_size: int = 4,
        num_workers: int = 0,
        bands: Sequence[str] = BurnIntensityNonGeo.all_band_names,
        train_transform: A.Compose | None | list[A.BasicTransform] = None,
        val_transform: A.Compose | None | list[A.BasicTransform] = None,
        test_transform: A.Compose | None | list[A.BasicTransform] = None,
        use_full_data: bool = True,
        no_data_replace: float | None = 0.0001,
        no_label_replace: int | None = -1,
        use_metadata: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(BurnIntensityNonGeo, batch_size, num_workers, **kwargs)
        self.data_root = data_root

        means = [MEANS[b] for b in bands]
        stds = [STDS[b] for b in bands]
        self.bands = bands
        self.train_transform = wrap_in_compose_is_list(train_transform)
        self.val_transform = wrap_in_compose_is_list(val_transform)
        self.test_transform = wrap_in_compose_is_list(test_transform)
        self.aug = Normalize(means, stds)
        self.use_full_data = use_full_data
        self.no_data_replace = no_data_replace
        self.no_label_replace = no_label_replace
        self.use_metadata = use_metadata

    def setup(self, stage: str) -> None:
        if stage in ["fit"]:
            self.train_dataset = self.dataset_class(
                split="train",
                data_root=self.data_root,
                transform=self.train_transform,
                bands=self.bands,
                use_full_data=self.use_full_data,
                no_data_replace=self.no_data_replace,
                no_label_replace=self.no_label_replace,
                use_metadata=self.use_metadata,
            )
        if stage in ["fit", "validate"]:
            self.val_dataset = self.dataset_class(
                split="val",
                data_root=self.data_root,
                transform=self.val_transform,
                bands=self.bands,
                use_full_data=self.use_full_data,
                no_data_replace=self.no_data_replace,
                no_label_replace=self.no_label_replace,
                use_metadata=self.use_metadata,
            )
        if stage in ["test"]:
            self.test_dataset = self.dataset_class(
                split="val",
                data_root=self.data_root,
                transform=self.test_transform,
                bands=self.bands,
                use_full_data=self.use_full_data,
                no_data_replace=self.no_data_replace,
                no_label_replace=self.no_label_replace,
                use_metadata=self.use_metadata,
            )
