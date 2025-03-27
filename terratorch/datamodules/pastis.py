import collections.abc
import re
from typing import Any

import albumentations as A  # noqa: N812
import torch
from torch.nn import functional as F

from terratorch.datamodules.utils import wrap_in_compose_is_list
from terratorch.datasets import PASTIS
from torchgeo.datamodules import NonGeoDataModule


def collate_fn(batch, pad_value=0):
    def pad_tensor(x, l, pad_value=0):
        padlen = l - x.shape[0]
        pad = [0 for _ in range(2 * len(x.shape[1:]))] + [0, padlen]
        return F.pad(x, pad=pad, value=pad_value)

    np_str_obj_array_pattern = re.compile(r"[SaUO]")
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if len(elem.shape) > 0:
            sizes = [e.shape[0] for e in batch]
            m = max(sizes)
            if not all(s == m for s in sizes):
                # pad tensors which have a temporal dimension
                batch = [pad_tensor(e, m, pad_value=pad_value) for e in batch]
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(batch, 0, out=out)
    elif (
        elem_type.__module__ == "numpy" and elem_type.__name__ not in ("str_", "string_")
    ):
        if elem_type.__name__ in ("ndarray", "memmap"):
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                msg = f"Format not managed : {elem.dtype}"
                raise TypeError(msg)

            return collate_fn([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, collections.abc.Mapping):
        return {key: collate_fn([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, "_fields"):  # namedtuple
        return elem_type(*(collate_fn(samples) for samples in zip(*batch, strict=False)))
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            msg = "each element in list of batch should be of equal size"
            raise RuntimeError(msg)
        transposed = zip(*batch, strict=False)
        return [collate_fn(samples) for samples in transposed]

    msg = f"Format not managed : {elem_type}"
    raise TypeError(msg)

class PASTISDataModule(NonGeoDataModule):
    def __init__(
        self,
        batch_size: int = 8,
        num_workers: int = 0,
        data_root: str = "./",
        truncate_image: int | None = None,
        pad_image: int | None = None,
        train_transform: A.Compose | None | list[A.BasicTransform] = None,
        val_transform: A.Compose | None | list[A.BasicTransform] = None,
        test_transform: A.Compose | None | list[A.BasicTransform] = None,
        use_dates: bool = True,
        utae: bool = False,
        use_resampled: bool = False,
        remove_cloudy_timesteps: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            PASTIS,
            batch_size=batch_size,
            num_workers=num_workers,
            **kwargs,
        )

        self.truncate_image = truncate_image
        self.pad_image = pad_image
        self.aug = lambda x: x
        self.train_transform = wrap_in_compose_is_list(train_transform)
        self.val_transform = wrap_in_compose_is_list(val_transform)
        self.test_transform = wrap_in_compose_is_list(test_transform)
        self.use_dates = use_dates
        self.utae = utae
        self.data_root = data_root
        self.kwargs = kwargs
        self.use_resampled = use_resampled
        self.remove_cloudy_timesteps = remove_cloudy_timesteps

    def setup(self, stage: str) -> None:
        if stage in ["fit"]:
            self.train_dataset = PASTIS(
                folds=[1,2,3],
                data_root=self.data_root,
                transform=self.train_transform,
                truncate_image=self.truncate_image,
                pad_image=self.pad_image,
                use_dates=self.use_dates,
                utae=self.utae,
                use_resampled=self.use_resampled,
                remove_cloudy_timesteps=self.remove_cloudy_timesteps,
                training=True,
                **self.kwargs,
            )
        if stage in ["fit", "validate"]:
            self.val_dataset = PASTIS(
                folds=[4],
                data_root=self.data_root,
                transform=self.val_transform,
                truncate_image=self.truncate_image,
                pad_image=self.pad_image,
                use_dates=self.use_dates,
                utae=self.utae,
                use_resampled=self.use_resampled,
                remove_cloudy_timesteps=self.remove_cloudy_timesteps,
                training=False,
                **self.kwargs,
            )
        if stage in ["test"]:
            self.test_dataset = PASTIS(
                folds=[5],
                data_root=self.data_root,
                transform=self.test_transform,
                truncate_image=self.truncate_image,
                pad_image=self.pad_image,
                use_dates=self.use_dates,
                utae=self.utae,
                use_resampled=self.use_resampled,
                remove_cloudy_timesteps=self.remove_cloudy_timesteps,
                training=False,
                **self.kwargs,
            )
