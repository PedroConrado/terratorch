import json
import os
import warnings
from datetime import datetime, timezone

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torchgeo.datasets import NonGeoDataset

from terratorch.datasets.utils import pad_dates_numpy, pad_numpy, default_transform


def load_cloudy_index(index_path):
    """
    Carrega o índice de timesteps nublados a partir de um arquivo JSON.
    """
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Arquivo de índice de nuvens não encontrado: {index_path}")
    with open(index_path, "r") as file:
        return json.load(file)

class PASTIS(NonGeoDataset):
    def __init__(
        self,
        data_root,
        norm=True,  # noqa: FBT002
        target="semantic",
        folds=None,
        reference_date="2018-09-01",
        date_interval = (-200,600),
        class_mapping=None,
        transform = None,
        truncate_image = None,
        pad_image = None,
        satellites=["S2"],  # noqa: B006
        use_dates = True,
        utae = False,
        use_resampled=False,
        remove_cloudy_timesteps=False,
        training=True,
    ):
        """
        Pytorch Dataset class to load samples from the PASTIS dataset, for semantic and
        panoptic segmentation.

        Args:
            data_root (str): Path to the dataset.
            norm (bool): If true, images are standardised using pre-computed
                channel-wise means and standard deviations.
            reference_date (str, Format : 'YYYY-MM-DD'): Defines the reference date
                based on which all observation dates are expressed. Along with the image
                time series and the target tensor, this dataloader yields the sequence
                of observation dates (in terms of number of days since the reference
                date). This sequence of dates is used for instance for the positional
                encoding in attention based approaches.
            target (str): 'semantic' or 'instance'. Defines which type of target is
                returned by the dataloader.
                * If 'semantic' the target tensor is a tensor containing the class of
                each pixel.
                * If 'instance' the target tensor is the concatenation of several
                signals, necessary to train the Parcel-as-Points module:
                    - the centerness heatmap,
                    - the instance ids,
                    - the voronoi partitioning of the patch with regards to the parcels'
                    centers,
                    - the (height, width) size of each parcel,
                    - the semantic label of each parcel,
                    - the semantic label of each pixel.
            folds (list, optional): List of ints specifying which of the 5 official
                folds to load. By default (when None is specified), all folds are loaded.
            class_mapping (dict, optional): A dictionary to define a mapping between the
                default 18 class nomenclature and another class grouping. If not provided, 
                the default class mapping is used.
            transform (callable, optional): A transform to apply to the loaded data 
                (images, dates, and masks). By default, no transformation is applied.
            truncate_image (int, optional): Truncate the time dimension of the image to 
                a specified number of timesteps. If None, no truncation is performed.
            pad_image (int, optional): Pad the time dimension of the image to a specified 
                number of timesteps. If None, no padding is applied.
            satellites (list): Defines the satellites to use. If you are using PASTIS-R, you
                have access to Sentinel-2 imagery and Sentinel-1 observations in Ascending
                and Descending orbits, respectively S2, S1A, and S1D. For example, use
                satellites=['S2', 'S1A'] for Sentinel-2 + Sentinel-1 ascending time series,
                or satellites=['S2', 'S1A', 'S1D'] to retrieve all time series. If you are using
                PASTIS, only S2 observations are available.
        """
        if target not in ["semantic", "instance"]:
            msg = f"Target '{target}' not recognized. Use 'semantic', or 'instance'."
            raise ValueError(msg)
        valid_satellites = {"S2", "S1A", "S1D"}
        for sat in satellites:
            if sat not in valid_satellites:
                msg = f"Satellite '{sat}' not recognized. Valid options are {valid_satellites}."
                raise ValueError(msg)

        super().__init__()
        self.data_root = data_root
        self.use_resampled = use_resampled
        self.data_dir = os.path.join(
            data_root, "DATA_S2_monthly_aggregated" if use_resampled else "DATA_S2"
        )
        metadata_file = os.path.join(
            data_root, "metadata_monthly_aggregated.geojson" if use_resampled else "metadata.geojson"
        )
        self.remove_cloudy_timesteps = remove_cloudy_timesteps
        self.cloudy_index_path = os.path.join(data_root, "cloudy_frames_index.json")
        if self.remove_cloudy_timesteps:
            self.cloudy_index = load_cloudy_index(self.cloudy_index_path)
        else:
            self.cloudy_index = None
        self.norm = norm
        self.reference_date = datetime(*map(int, reference_date.split("-")), tzinfo=timezone.utc)
        self.class_mapping = (
            np.vectorize(lambda x: class_mapping[x])
            if class_mapping is not None
            else class_mapping
        )
        self.target = target
        self.satellites = satellites
        self.transform = transform if transform else default_transform
        self.truncate_image = truncate_image
        self.pad_image = pad_image
        self.utae = utae
        self.training = training
        # loads patches metadata
        self.meta_patch = gpd.read_file(metadata_file)
        self.meta_patch.index = self.meta_patch["ID_PATCH"].astype(int)
        self.meta_patch.sort_index(inplace=True)
        # stores table for each satalite date
        self.date_tables = {s: None for s in satellites}
        # date interval used in the PASTIS benchmark paper.
        date_interval_begin, date_interval_end = date_interval
        self.date_range = np.array(range(date_interval_begin, date_interval_end))
        for s in satellites:
            # maps patches to its observation dates
            dates = self.meta_patch[f"dates-{s}"]
            date_table = pd.DataFrame(
                index=self.meta_patch.index, columns=self.date_range, dtype=int
            )
            for pid, date_seq in dates.items():
                if type(date_seq) is str:
                    date_seq = json.loads(date_seq)  # noqa: PLW2901
                # convert date to days since obersavation format
                d = pd.DataFrame().from_dict(date_seq, orient="index")
                d = d[0].apply(
                    lambda x: (
                        datetime(int(str(x)[:4]), int(str(x)[4:6]), int(str(x)[6:]), tzinfo=timezone.utc)
                        - self.reference_date
                    ).days
                )
                date_table.loc[pid, d.values] = 1
            date_table = date_table.fillna(0)
            self.date_tables[s] = {
                index: np.array(list(d.values()))
                for index, d in date_table.to_dict(orient="index").items()
            }

        # selects patches correspondig to selected folds
        if folds is not None:
            self.meta_patch = pd.concat(
                [self.meta_patch[self.meta_patch["Fold"] == f] for f in folds]
            )

        self.len = self.meta_patch.shape[0]
        self.id_patches = self.meta_patch.index

        # loads normalization values
        if norm:
            self.norm = {}
            for s in self.satellites:
                with open(
                    os.path.join(data_root, f"NORM_{s}_patch.json")
                ) as file:
                    normvals = json.loads(file.read())
                selected_folds = folds if folds is not None else range(1, 6)
                means = [normvals[f"Fold_{f}"]["mean"] for f in selected_folds]
                stds = [normvals[f"Fold_{f}"]["std"] for f in selected_folds]
                self.norm[s] = np.stack(means).mean(axis=0), np.stack(stds).mean(axis=0)
                self.norm[s] = (
                    self.norm[s][0],
                    self.norm[s][1],
                )
        else:
            self.norm = None

        self.use_dates = use_dates

    def __len__(self):
        return self.len

    def get_dates(self, id_patch, sat):
        if self.use_resampled:
            return np.array(list(self.meta_patch.loc[id_patch, f"dates-{sat}"].values()))
        else:
            return self.date_range[np.where(self.date_tables[sat][id_patch] == 1)[0]]

    def __getitem__(self, item):
        id_patch = self.id_patches[item]
        output = {}
        satellites = {}
        dates = {}
        for satellite in self.satellites:
            data = np.load(
                os.path.join(self.data_dir, f"{satellite}_{id_patch}.npy")
            ).astype(np.float32)

            if self.norm is not None:
                    data = data - self.norm[satellite][0][None, :, None, None]
                    data = data / self.norm[satellite][1][None, :, None, None]

            date = np.array(self.get_dates(id_patch, satellite))
            if self.remove_cloudy_timesteps and str(f"{satellite}_{id_patch}.npy") in self.cloudy_index:
                cloudy_indices = np.array(self.cloudy_index[f"{satellite}_{id_patch}.npy"])
                if len(cloudy_indices) > 0:
                    mask = np.ones(data.shape[0], dtype=bool)
                    mask[cloudy_indices] = False
                    data = data[mask]
                    date = date[mask]

            if data.shape[0] == 0:
                msg = f"All timesteps removed for {id_patch}"
                raise ValueError(msg)

            if self.truncate_image and data.shape[0] > self.truncate_image:
                if self.training:
                    bins = np.linspace(0, data.shape[0], self.truncate_image + 1, endpoint=True).astype(int)
                    indices = []
                    for i in range(self.truncate_image):
                        start = bins[i]
                        end = bins[i+1]
                        if start < end:
                            idx = np.random.randint(start, end)
                        else:
                            idx = start
                        indices.append(idx)
                    indices = np.array(indices)
                else:
                    indices = np.linspace(0, data.shape[0] - 1, self.truncate_image, dtype=int)
                data = data[indices]
                date = date[indices]

            if self.pad_image and data.shape[0] < self.pad_image:
                data = pad_numpy(data, self.pad_image)
                date = pad_dates_numpy(date, self.pad_image)

            satellites[satellite] = data.astype(np.float32)
            dates[satellite] = torch.from_numpy(date)

        target = np.load(
            os.path.join(self.data_root, "ANNOTATIONS", f"TARGET_{id_patch}.npy")
        )
        target = target[0].astype(int)
        if self.class_mapping is not None:
            target = self.class_mapping(target)

        output["image"] = satellites["S2"].transpose(0, 2, 3, 1)
        output["mask"] = target
        if self.transform:
            output = self.transform(**output)
            if not self.utae:
                output["image"] = output["image"][[0, 1, 2, 7, 8, 9], :]
            else:
                output["image"] = output["image"].permute(1, 0, 2, 3)

        if self.use_dates:
            output["batch_positions"] = dates["S2"]

        output["mask"] = output["mask"].long()

        return output


    def plot(self, sample, suptitle=None):
        target = sample["mask"]

        image_data = sample["image"]

        if torch.is_tensor(image_data):
            image_data = image_data.numpy()
        rgb_images = []
        for i in range(image_data.shape[1]):
            rgb_image = image_data[[2,1,0], i, :, :].transpose(1, 2, 0)
            rgb_min = rgb_image.min(axis=(0, 1), keepdims=True)
            rgb_max = rgb_image.max(axis=(0, 1), keepdims=True)
            denom = rgb_max - rgb_min
            denom[denom == 0] = 1
            rgb_image = (rgb_image - rgb_min) / denom

            rgb_images.append(np.clip(rgb_image, 0, 1))

        return self._plot_sample(rgb_images, target, suptitle=suptitle)

    def _plot_sample(
        self,
        images: list[np.ndarray],
        target: torch.Tensor | None,
        suptitle: str | None = None
    ):
        num_images = len(images)
        cols = 5
        rows = (num_images + cols) // cols

        fig, ax = plt.subplots(rows, cols, figsize=(20, 4 * rows))

        for i, image in enumerate(images):
            ax[i // cols, i % cols].imshow(image)
            ax[i // cols, i % cols].set_title(f"Image {i + 1}")
            ax[i // cols, i % cols].axis("off")

        if target is not None:
            if rows * cols > num_images:
                target_ax = ax[(num_images) // cols, (num_images) % cols]
            else:
                fig.add_subplot(rows + 1, 1, 1)
                target_ax = fig.gca()

            target_ax.imshow(target.numpy(), cmap="tab20")
            target_ax.set_title("Mask")
            target_ax.axis("off")

        for k in range(num_images + 1, rows * cols):
            ax[k // cols, k % cols].axis("off")

        if suptitle:
            plt.suptitle(suptitle)

        plt.tight_layout()
        return fig
