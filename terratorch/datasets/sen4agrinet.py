import random
import warnings
from pathlib import Path

import albumentations as A
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import xarray as xr
from einops import rearrange
from einops import rearrange
import matplotlib.pyplot as plt
from terratorch.datasets.utils import default_transform
from torchgeo.datasets import NonGeoDataset

# PERMANECEM OS MESMOS
CAT_TILES = ["31TBF", "31TCF", "31TCG", "31TDF", "31TDG"]
FR_TILES = ["31TCJ", "31TDK", "31TCL", "31TDM", "31UCP", "31UDR"]
SELECTED_CLASSES = [110,120,140,150,160,170,330,435,438,510,770]
BANDS = {
    "B01": 60,
    "B02": 10,
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

MONTH_START = 3  # Abril (índice 3)
MONTH_END   = 9  # Setembro (índice 9 para slice)
GROUP_FREQ  = "1MS"

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
        unflatten_on_plot: bool = True,
    ):
        super().__init__()

        self.data_root = Path(data_root) / "data"
        self.transform = transform if transform else default_transform
        self.scenario = scenario
        self.split = split
        self.seed = seed
        if bands is None:
            bands = ["B01","B02","B03","B04","B05","B06","B07","B08","B09","B10","B11","B12","B8A"]
        self.bands = sorted(bands)
        self.requires_norm = requires_norm
        self.binary_labels = binary_labels
        self.unflatten_on_plot = unflatten_on_plot

        encoder = {val: i+1 for i, val in enumerate(sorted(SELECTED_CLASSES))}
        encoder[0] = 0
        self.linear_encoder = encoder


        # Split files
        all_files = list((Path(data_root)/"data").glob("**/*.nc"))
        self.train_files, self.val_files, self.test_files = self.split_data(all_files)
        if split == "train":
            self.image_files = self.train_files
        elif split == "val":
            self.image_files = self.val_files
        else:
            self.image_files = self.test_files

    def split_data(self, all_files):
        random.seed(self.seed)
        if self.scenario == "random":
            random.shuffle(all_files)
            total_files = len(all_files)
            train_split = int(0.6 * total_files)
            val_split = int(0.8 * total_files)
            train_files = all_files[:train_split]
            val_files = all_files[train_split:val_split]
            test_files = all_files[val_split:]
        elif self.scenario == "spatial":
            cat_files = [f for f in all_files if any(tile in f.stem for tile in CAT_TILES)]
            fr_files  = [f for f in all_files if any(tile in f.stem for tile in FR_TILES)]
            val_split_cat = int(0.2 * len(cat_files))
            train_files = cat_files[val_split_cat:]
            val_files  = cat_files[:val_split_cat]
            test_files = fr_files
        elif self.scenario == "spatio-temporal":
            fr_files  = [f for f in all_files if any(tile in f.stem for tile in FR_TILES)]
            cat_files = [f for f in all_files if any(tile in f.stem for tile in CAT_TILES)]
            fr_2019 = [f for f in fr_files  if "2019" in f.stem]
            cat_2020= [f for f in cat_files if "2020" in f.stem]
            val_split_fr_2019 = int(0.2 * len(fr_2019))
            train_files = fr_2019[val_split_fr_2019:]
            val_files  = fr_2019[:val_split_fr_2019]
            test_files = cat_2020
        return train_files, val_files, test_files

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx: int):
        nc_file = self.image_files[idx]
        with h5py.File(nc_file, "r") as f:
            data_list = []
            time_data = None
            for b in self.bands:
                arr = f[b][b][:]
                t = f[b]["time"][:]
                idxs = np.argsort(t)
                arr = arr[idxs].astype(np.float32)
                t = t[idxs]
                ratio = int(BANDS[b]/BANDS[REFERENCE_BAND])
                if ratio != 1:
                    arr = np.repeat(arr, ratio, axis=1)
                    arr = np.repeat(arr, ratio, axis=2)
                if time_data is None:
                    time_data = pd.to_datetime(t, unit="s", origin="1970-01-01")
                data_list.append(arr)

            stacked = np.stack(data_list, axis=0)
            da = xr.DataArray(stacked, dims=["bands","time","y","x"],
                              coords={"time": time_data, "bands": self.bands})
            da = da.resample(time=GROUP_FREQ).median(dim="time")
            da = da.interpolate_na(dim="time", method="linear", fill_value="extrapolate")

            if da.time.size < 12:
                missing = 12 - da.time.size
                if missing > 0:
                    last_t = da.time[-1].values
                    extra  = pd.date_range(start=last_t, periods=missing+1, freq=GROUP_FREQ)[1:]
                    da = da.reindex(time=da.time.to_index().union(extra), method="ffill")

            sub_da = da.isel(time=slice(MONTH_START, MONTH_END))
            arr_out = sub_da.values
            if self.requires_norm:
                arr_out = arr_out / NORMALIZATION_DIV

            labels = f["labels"]["labels"][:].astype(np.int64)
            new_labels = np.zeros_like(labels)
            for cid, lid in self.linear_encoder.items():
                new_labels[labels == cid] = lid
            if self.binary_labels:
                new_labels[new_labels != 0] = 1
            arr_out = np.moveaxis(arr_out, 0, -1)
            output = {
                "image": arr_out.astype(np.float32),
                "mask": new_labels.astype(np.int64),
            }
            if self.transform:
                output = self.transform(**output)
            output["mask"] = output["mask"].long()
            return output

    def plot(self, sample, suptitle=None):
        rgb_bands = ["B04", "B03", "B02"]
        if not all(b in BANDS for b in rgb_bands):
            warnings.warn("RGB bands not available.")
            return
        images = sample["image"]
        labels = sample.get("mask", None)

        if self.unflatten_on_plot:
            n_bands = len(self.bands)
            n_time  = 6
            images  = rearrange(images, "(t b) h w -> b t h w", t=n_time, b=n_bands)

        sorted_b = sorted(self.bands)
        idx_b02  = sorted_b.index("B02")
        idx_b03  = sorted_b.index("B03")
        idx_b04  = sorted_b.index("B04")

        rgb_images = []
        for t in range(images.shape[1]):
            rgb_slice = images[[idx_b04,idx_b03,idx_b02], t, :, :]
            mn = rgb_slice.min()
            mx = rgb_slice.max()
            denom = mx - mn if mx>mn else 1
            rgb_slice = (rgb_slice - mn)/denom
            rgb_slice = np.clip(rgb_slice,0,1)
            rgb_images.append(rgb_slice)

        num_img = len(rgb_images)
        cols = 5
        rows = (num_img+cols-1)//cols
        fig, ax = plt.subplots(rows,cols,figsize=(20,4*rows))

        if rows==1 and cols==1:
            ax = np.array([[ax]])
        elif rows==1:
            ax = ax[np.newaxis,:]
        elif cols==1:
            ax = ax[:,np.newaxis]

        for i,img_ in enumerate(rgb_images):
            rr,cc = i//cols, i%cols
            ax[rr,cc].imshow(np.moveaxis(img_.numpy(),0,-1))
            ax[rr,cc].axis("off")
            ax[rr,cc].set_title(f"T{i+1}")

        if labels is not None:
            if rows*cols>num_img:
                targ_ax = ax[num_img//cols, num_img%cols]
            else:
                fig.add_subplot(rows+1,1,1)
                targ_ax = fig.gca()
            targ_ax.imshow(labels, cmap="tab20")
            targ_ax.set_title("Labels")
            targ_ax.axis("off")

        for k in range(num_img, rows*cols):
            ax[k//cols,k%cols].axis("off")

        if suptitle:
            plt.suptitle(suptitle)
        plt.tight_layout()
        plt.show()
