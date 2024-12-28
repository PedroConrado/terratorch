from typing import Any

import albumentations as A  # noqa: N812

from terratorch.datamodules.utils import wrap_in_compose_is_list
from terratorch.datasets import Sen4AgriNet
from torchgeo.datamodules import NonGeoDataModule


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
        seed: int = 42,
        scenario: str = "random",
        requires_norm: bool = True,
        binary_labels: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            Sen4AgriNet,
            batch_size=batch_size,
            num_workers=num_workers,
            **kwargs,
        )
        self.bands = bands
        self.seed = seed
        self.train_transform = wrap_in_compose_is_list(train_transform)
        self.val_transform = wrap_in_compose_is_list(val_transform)
        self.test_transform = wrap_in_compose_is_list(test_transform)
        self.data_root = data_root
        self.scenario = scenario
        self.aug = lambda x: x
        self.requires_norm = requires_norm
        self.binary_labels = binary_labels
        self.kwargs = kwargs

    def setup(self, stage: str) -> None:
        if stage in ["fit"]:
            self.train_dataset = Sen4AgriNet(
                split="train",
                data_root=self.data_root,
                transform=self.train_transform,
                bands=self.bands,
                seed=self.seed,
                scenario=self.scenario,
                requires_norm=self.requires_norm,
                binary_labels=self.binary_labels,
                **self.kwargs,
            )
        if stage in ["fit", "validate"]:
            self.val_dataset = Sen4AgriNet(
                split="val",
                data_root=self.data_root,
                transform=self.val_transform,
                bands=self.bands,
                seed=self.seed,
                scenario=self.scenario,
                requires_norm=self.requires_norm,
                binary_labels=self.binary_labels,
                **self.kwargs,
            )
        if stage in ["test"]:
            self.test_dataset = Sen4AgriNet(
                split="test",
                data_root=self.data_root,
                transform=self.test_transform,
                bands=self.bands,
                seed=self.seed,
                scenario=self.scenario,
                requires_norm=self.requires_norm,
                binary_labels=self.binary_labels,
                **self.kwargs,
            )
