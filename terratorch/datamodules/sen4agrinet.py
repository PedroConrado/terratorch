from typing import Any

import albumentations as A  # noqa: N812
from torchgeo.datamodules import NonGeoDataModule
from torchgeo.transforms import AugmentationSequential

from terratorch.datamodules.utils import wrap_in_compose_is_list
from terratorch.datasets import Sen4AgriNet


class Sen4AgriNetDataModule(NonGeoDataModule):
    def __init__(
        self,
        bands: list[str] | None = None,
        batch_size: int = 8,
        num_workers: int = 0,
        data_root: str = "./",
        train_transform: A.Compose | None | list[A.BasicTransform] = None,
        val_transform: A.Compose | None | list[A.BasicTransform] = None,
        test_transform: A.Compose | None | list[A.BasicTransform] = None,
        truncate_image: int | None = 4,
        pad_image: int | None = 4,
        interpolate_and_concat_bands: bool = True,  # noqa: FBT002, FBT001
        seed: int = 42,
        aug: AugmentationSequential = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            Sen4AgriNet,
            batch_size=batch_size,
            num_workers=num_workers,
            aug=aug,
            **kwargs,
        )
        self.bands = bands
        self.truncate_image = truncate_image
        self.pad_image = pad_image
        self.interpolate_and_concat_bands = interpolate_and_concat_bands
        self.seed = seed
        self.train_transform = wrap_in_compose_is_list(train_transform)
        self.val_transform = wrap_in_compose_is_list(val_transform)
        self.test_transform = wrap_in_compose_is_list(test_transform)
        self.data_root = data_root
        self.kwargs = kwargs


    def setup(self, stage: str) -> None:
        if stage in ["fit"]:
            self.train_dataset = Sen4AgriNet(
                split="train",
                data_root=self.data_root,
                transform=self.train_transform,
                bands=self.bands,
                truncate_image = self.truncate_image,
                pad_image = self.pad_image,
                interpolate_and_concat_bands = self.interpolate_and_concat_bands,
                seed = self.seed,
                **self.kwargs,
            )
        if stage in ["fit", "validate"]:
            self.val_dataset = Sen4AgriNet(
                split="val",
                data_root=self.data_root,
                transform=self.val_transform,
                bands=self.bands,
                truncate_image = self.truncate_image,
                pad_image = self.pad_image,
                interpolate_and_concat_bands = self.interpolate_and_concat_bands,
                seed = self.seed,
                **self.kwargs,
            )
        if stage in ["test"]:
            self.test_dataset = Sen4AgriNet(
                split="test",
                data_root=self.data_root,
                transform=self.test_transform,
                bands=self.bands,
                truncate_image = self.truncate_image,
                pad_image = self.pad_image,
                interpolate_and_concat_bands = self.interpolate_and_concat_bands,
                seed = self.seed,
                **self.kwargs,
            )