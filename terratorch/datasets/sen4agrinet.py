import random
import warnings
from pathlib import Path

import albumentations as A  # noqa: N812
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import xarray as xr

from terratorch.datasets.utils import default_transform
from torchgeo.datasets import NonGeoDataset

CAT_TILES = ["31TBF", "31TCF", "31TCG", "31TDF", "31TDG"]
FR_TILES = ["31TCJ", "31TDK", "31TCL", "31TDM", "31UCP", "31UDR"]

MAX_TEMPORAL_IMAGE_SIZE = (366,366)

SELECTED_CLASSES = [
    110,
    120,
    140,
    150,
    160,
    170,
    330,
    435,
    438,
    510,
    770,
]

BANDS = {
    "B01": 60,
    "B02": 10,  # reference band
    "B03": 10,
    "B04": 10,
    "B05": 20,
    "B06": 20,
    "B07": 20,
    "B08": 10,
    "B09": 60,
    "B10": 60,
    "B11": 20,
    "B12": 20,
    "B8A": 20
}
REFERENCE_BAND = "B02"
NORMALIZATION_DIV = 10000.0
WINDOW_LEN = 6
GROUP_FREQ = "1MS"

class Sen4AgriNet(NonGeoDataset):
    def __init__(
        self,
        data_root: str,
        bands: list[str] | None = None,
        scenario: str = "random",
        split: str = "train",
        transform: A.Compose = None,
        seed: int = 42,
        requires_norm: bool = True,
        binary_labels: bool = False,
        linear_encoder: dict = None,
    ):
        self.data_root = Path(data_root) / "data"
        self.transform = transform if transform else default_transform
        self.scenario = scenario
        self.seed = seed
        if bands is None:
            bands = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B09", "B10", "B11", "B12", "B8A"]
        self.bands = sorted(bands)
        self.requires_norm = requires_norm
        self.binary_labels = binary_labels
        if linear_encoder is None:
            encoder = {val: i+1 for i, val in enumerate(sorted(SELECTED_CLASSES))}
            encoder[0] = 0
            self.linear_encoder = encoder
        else:
            self.linear_encoder = linear_encoder
        self.image_files = list(self.data_root.glob("**/*.nc"))
        self.train_files, self.val_files, self.test_files = self.split_data()
        if split == "train":
            self.image_files = self.train_files
        elif split == "val":
            self.image_files = self.val_files
        elif split == "test":
            self.image_files = self.test_files

    def __len__(self):
        return len(self.image_files)

    def split_data(self):
        random.seed(self.seed)
        if self.scenario == "random":
            random.shuffle(self.image_files)
            total_files = len(self.image_files)
            train_split = int(0.6 * total_files)
            val_split = int(0.8 * total_files)
            train_files = self.image_files[:train_split]
            val_files = self.image_files[train_split:val_split]
            test_files = self.image_files[val_split:]
        elif self.scenario == "spatial":
            catalonia_files = [f for f in self.image_files if any(tile in f.stem for tile in CAT_TILES)]
            france_files = [f for f in self.image_files if any(tile in f.stem for tile in FR_TILES)]
            val_split_cat = int(0.2 * len(catalonia_files))
            train_files = catalonia_files[val_split_cat:]
            val_files = catalonia_files[:val_split_cat]
            test_files = france_files
        elif self.scenario == "spatio-temporal":
            france_files = [f for f in self.image_files if any(tile in f.stem for tile in FR_TILES)]
            catalonia_files = [f for f in self.image_files if any(tile in f.stem for tile in CAT_TILES)]
            france_2019_files = [f for f in france_files if "2019" in f.stem]
            catalonia_2020_files = [f for f in catalonia_files if "2020" in f.stem]
            val_split_france_2019 = int(0.2 * len(france_2019_files))
            train_files = france_2019_files[val_split_france_2019:]
            val_files = france_2019_files[:val_split_france_2019]
            test_files = catalonia_2020_files
        return train_files, val_files, test_files

    def __getitem__(self, index: int):
        patch_file = self.image_files[index]
        with h5py.File(patch_file, "r") as f:
            bands_data = []
            time_data = None
            for b in self.bands:
                bd = f[b][b][:]
                t = f[b]["time"][:]
                idxs = np.argsort(t)
                bd = bd[idxs].astype(np.float32)
                t = t[idxs]
                # Replicate pixels for different resolution bands
                expand_ratio = int(BANDS[b] / BANDS[REFERENCE_BAND])
                if expand_ratio != 1:
                    bd = np.repeat(bd, expand_ratio, axis=1)
                    bd = np.repeat(bd, expand_ratio, axis=2)
                if time_data is None:
                    time_data = pd.to_datetime(t, unit="s", origin="1970-01-01")
                bands_data.append(bd)
            bands_data = np.stack(bands_data, axis=0)
            da = xr.DataArray(bands_data, dims=["bands","time","y","x"], coords={"time": time_data, "bands":self.bands})
            da_med = da.resample(time=GROUP_FREQ).median(dim="time")
            # Interpolate missing months
            da_med = da_med.interpolate_na(dim="time", method="linear", fill_value="extrapolate")
            if da_med.time.size < WINDOW_LEN:
                # Ensuring exactly 6 months
                da_med = da_med.reindex(time=da_med.time.to_index().union(pd.date_range(start=da_med.time.values[0], periods=WINDOW_LEN, freq=GROUP_FREQ)), method="ffill")
            da_med = da_med.isel(time=slice(0, WINDOW_LEN))
            medians = da_med.values
            if self.requires_norm:
                medians = medians / NORMALIZATION_DIV
            labels = f["labels"]["labels"][:].astype(np.int64)
            _labels = np.zeros_like(labels)
            for cid, lid in self.linear_encoder.items():
                _labels[labels == cid] = lid
            labels = _labels
            if self.binary_labels:
                labels[labels!=0] = 1
            medians = np.moveaxis(medians, 0, -1)
            output = {
                "image": medians.astype(np.float32),
                "mask": labels.astype(np.int64),
                "idx": index
            }
            if self.transform:
                output = self.transform(**output)
            return output

    def plot(self, sample, suptitle=None):
        rgb_bands = ["B04", "B03", "B02"]
        if not all(b in self.bands for b in rgb_bands):
            warnings.warn("RGB bands not available.")  # noqa: B028
            return None
        idx_b02 = self.bands.index("B02")
        idx_b03 = self.bands.index("B03")
        idx_b04 = self.bands.index("B04")

        images = sample["image"]
        labels = sample.get("labels", None)
        rgb_images = []

        for t in range(images.shape[1]):
            rgb_image = images[[idx_b04, idx_b03, idx_b02], t, :, :]
            rgb_min = rgb_image.min()
            rgb_max = rgb_image.max()
            denom = rgb_max - rgb_min
            if denom == 0:
                denom = 1
            rgb_image = (rgb_image - rgb_min) / denom
            rgb_image = np.clip(rgb_image, 0, 1)
            rgb_image = np.transpose(rgb_image, (1, 2, 0))
            rgb_images.append(rgb_image)

        num_images = len(rgb_images)
        cols = 5
        rows = (num_images + cols - 1) // cols
        fig, ax = plt.subplots(rows, cols, figsize=(20, 4 * rows))

        if rows == 1 and cols == 1:
            ax = np.array([[ax]])
        elif rows == 1:
            ax = ax[np.newaxis, :]
        elif cols == 1:
            ax = ax[:, np.newaxis]

        for i, image in enumerate(rgb_images):
            ax[i // cols, i % cols].imshow(image)
            ax[i // cols, i % cols].set_title(f"T{i+1}")
            ax[i // cols, i % cols].axis("off")

        if labels is not None:
            if rows * cols > num_images:
                target_ax = ax[(num_images) // cols, (num_images) % cols]
            else:
                fig.add_subplot(rows + 1, 1, 1)
                target_ax = fig.gca()
            target_ax.imshow(labels, cmap="tab20")
            target_ax.set_title("Labels")
            target_ax.axis("off")

        for k in range(num_images, rows * cols):
            ax[k // cols, k % cols].axis("off")

        if suptitle:
            fig.suptitle(suptitle)

        plt.tight_layout()
        return fig

    def map_mask_to_discrete_classes(self, mask, encoder):
        map_func = np.vectorize(lambda x: encoder.get(x, 0))
        return map_func(mask)
