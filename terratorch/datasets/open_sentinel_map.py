import json
import random
import re
from pathlib import Path

import albumentations as A  # noqa: N812
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F  # noqa: N812
from matplotlib.figure import Figure
from PIL import Image
from torch import Tensor

from terratorch.datasets.utils import default_transform, pad_numpy
from torchgeo.datasets import NonGeoDataset

MAX_TEMPORAL_IMAGE_SIZE = (192, 192)


class OpenSentinelMap(NonGeoDataset):
    def __init__(
        self,
        data_root: str,
        split: str = "train",
        bands: list[str] | None = None,
        transform: A.Compose | None = None,
        spatial_interpolate_and_stack_temporally: bool = True,
        pad_image: int | None = None,
        truncate_image: int | None = None,
    ) -> None:
        """

        Args:
            data_root (str): Path to dataset root.
            split (str): 'train', 'val', or 'test'.
            bands (list[str], optional): Bands to load. Default: ['gsd_10', 'gsd_20', 'gsd_60'].
            transform (A.Compose, optional): Transformations to apply.
            spatial_interpolate_and_stack_temporally (bool): If True, interpolate all bands to same resolution and return a single array.
            pad_image (int, optional): Temporal padding.
            truncate_image (int, optional): Temporal truncation.
        """
        if bands is None:
            bands = ["gsd_10", "gsd_20", "gsd_60"]

        allowed_bands = {"gsd_10", "gsd_20", "gsd_60"}
        for band in bands:
            if band not in allowed_bands:
                msg = f"Band '{band}' not recognized. Available: {', '.join(allowed_bands)}"
                raise ValueError(msg)

        if split not in ["train", "val", "test"]:
            msg = f"Split '{split}' not recognized."
            raise ValueError(msg)

        self.data_root = Path(data_root)
        split_mapping = {"train": "training", "val": "validation", "test": "testing"}
        split_folder = split_mapping[split]
        self.imagery_root = self.data_root / "osm_sentinel_imagery"
        self.label_root = self.data_root / "osm_label_images_v10"
        self.auxiliary_data = pd.read_csv(self.data_root / "spatial_cell_info.csv")
        self.auxiliary_data = self.auxiliary_data[self.auxiliary_data["split"] == split_folder]
        self.bands = bands
        self.transform = transform if transform else default_transform
        self.label_mappings = self._load_label_mappings()
        self.split_data = self.auxiliary_data[self.auxiliary_data["split"] == split_folder]
        self.spatial_interpolate_and_stack_temporally = spatial_interpolate_and_stack_temporally
        self.pad_image = pad_image
        self.truncate_image = truncate_image

        self.samples = []
        for _, row in self.split_data.iterrows():
            mgrs_tile = row["MGRS_tile"]
            spatial_cell = str(row["cell_id"])
            label_file = self.label_root / mgrs_tile / f"{spatial_cell}.png"
            if label_file.exists():
                self.samples.append((mgrs_tile, spatial_cell))

    def _load_label_mappings(self):
        with open(self.data_root / "osm_categories.json") as f:
            return json.load(f)

    def _extract_date_from_filename(self, filename: str) -> str:
        match = re.search(r"(\d{8})", filename)
        if match:
            return match.group(1)
        else:
            msg = f"Date not found in filename {filename}"
            raise ValueError(msg)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        mgrs_tile, spatial_cell = self.samples[index]
        spatial_cell_path = self.imagery_root / mgrs_tile / spatial_cell

        npz_files = list(spatial_cell_path.glob("*.npz"))
        npz_files.sort(key=lambda x: self._extract_date_from_filename(x.stem))

        if self.spatial_interpolate_and_stack_temporally:
            images_over_time = []
            for npz_file in npz_files:
                data = np.load(npz_file)
                interpolated_bands = []
                for band in self.bands:
                    band_frame = data[band].astype(np.float32)  # [H, W, C]
                    band_frame = torch.from_numpy(band_frame).permute(2, 0, 1)  # [C, H, W]
                    interpolated = F.interpolate(
                        band_frame.unsqueeze(0),
                        size=MAX_TEMPORAL_IMAGE_SIZE,
                        mode="bilinear",
                        align_corners=False
                    ).squeeze(0)  # [C, H, W]
                    interpolated_bands.append(interpolated)
                concatenated_bands = torch.cat(interpolated_bands, dim=0)  # [C, H, W]
                images_over_time.append(concatenated_bands)

            images = torch.stack(images_over_time, dim=0).numpy()  # [T, C, H, W]

            if self.truncate_image:
                images = images[-self.truncate_image:]
            if self.pad_image:
                images = pad_numpy(images, self.pad_image)

            images = images.transpose(0, 2, 3, 1)  # [T,H,W,C]
            output = {"image": images}

        else:
            image_dict = {band: [] for band in self.bands}
            for npz_file in npz_files:
                data = np.load(npz_file)
                for band in self.bands:
                    band_frames = data[band].astype(np.float32)  # [H, W, C]
                    band_frames = np.transpose(band_frames, (2, 0, 1))  # [C,H,W]
                    image_dict[band].append(band_frames)

            # Truncate/Padding
            final_image_dict = {}
            for band in self.bands:
                band_images = image_dict[band]  # list of [C,H,W]
                band_images = np.stack(band_images, axis=0)  # [T,C,H,W]
                if self.truncate_image:
                    band_images = band_images[-self.truncate_image:]
                if self.pad_image:
                    band_images = pad_numpy(band_images, self.pad_image)
                final_image_dict[band] = band_images

            output = {"image": final_image_dict}

        # Loads MASK
        label_file = self.label_root / mgrs_tile / f"{spatial_cell}.png"
        mask = np.array(Image.open(label_file)).astype(int)  # [H,W]
        # Ignore UNALABEL/NONE.
        output["mask"] = mask  # [H,W]
        if self.transform:
            output = self.transform(**output)

        return output

    def plot(self, sample: dict[str, Tensor], suptitle: str | None = None) -> Figure:
        image = sample["image"]
        mask = sample["mask"]

        if mask.ndim == 3 and mask.shape[0] == 1:
            mask = mask.squeeze(0)

        if isinstance(image, dict):
            if "gsd_10" in image:
                img_tensor = image["gsd_10"]
            else:
                first_band = next(iter(image.keys()))
                img_tensor = image[first_band]

            img_tensor = img_tensor.cpu().numpy()
        else:
            img_tensor = image.cpu().numpy()

        C, T, H, W = img_tensor.shape

        fig, axes = plt.subplots(1, T+1, figsize=(4*(T+1), 4))
        if T+1 == 1:
            axes = [axes]

        for i in range(T):
            frame = img_tensor[:,i, :, :]  # [C,H,W]
            channels = min(C, 3)
            frame_rgb = frame[:channels]  # [channels,H,W]
            frame_rgb_min = frame_rgb.min()
            frame_rgb_max = frame_rgb.max()
            if frame_rgb_max > frame_rgb_min:
                frame_rgb = (frame_rgb - frame_rgb_min) / (frame_rgb_max - frame_rgb_min)
            frame_rgb = np.transpose(frame_rgb, (1, 2, 0))  # [H,W,C]
            axes[i].imshow(frame_rgb)
            axes[i].set_title(f"Frame {i+1}")
            axes[i].axis("off")

        mask_np = mask.cpu().numpy()
        axes[-1].imshow(mask_np, cmap="gray")
        axes[-1].set_title("Mask")
        axes[-1].axis("off")

        if suptitle is not None:
            plt.suptitle(suptitle)

        plt.tight_layout()
        return fig
