# from collections.abc import Sequence
# from typing import Any

# import albumentations as A
# import kornia.augmentation as K  # noqa: N812

# from terratorch.datamodules.utils import wrap_in_compose_is_list
# from terratorch.datasets import BioMasstersNonGeo
# from torchgeo.datamodules import NonGeoDataModule
# from torchgeo.transforms import AugmentationSequential

# # AGBM_MEAN = 63.4584
# # AGBM_STD = 72.21242

# # S1_MEANS = [
# #       0.08871397,
# #       0.02172604,
# #       0.08556002,
# #       0.02795591,
# #       0.75507677,
# #       0.6600374 
# #       ]
# # S1_STDS = [
# #       0.16714208,
# #       0.04876742,
# #       0.19260046,
# #       0.10272296,
# #       0.24945821,
# #       0.3590119
# #       ]
# # S2_MEANS = [
# #       1633.0802,
# #       1610.0035,
# #       1599.557,
# #       1916.7083,
# #       2478.8325,
# #       2591.326,
# #       2738.5837,
# #       2685.8281,
# #       1023.90204,
# #       696.48755,
# #       21.177078
# #       ]
# # S2_STDS = [
# #       2499.7146,
# #       2308.5298,
# #       2388.2268,
# #       2389.6375,
# #       2209.6467,
# #       2104.572,
# #       2194.209,
# #       2031.7762,
# #       934.0556,
# #       759.8444,
# #       49.352486]

# class BioMasstersNonGeoDataModule(NonGeoDataModule):
#     """NonGeo datamodule implementation for BioMassters."""

#     default_metadata_filename = "The_BioMassters_-_features_metadata.csv.csv"

#     def __init__(
#         self,
#         data_root: str,
#         batch_size: int = 4,
#         num_workers: int = 0,
#         bands: Sequence[str] = BioMasstersNonGeo.all_band_names,
#         train_transform: A.Compose | None | list[A.BasicTransform] = None,
#         val_transform: A.Compose | None | list[A.BasicTransform] = None,
#         test_transform: A.Compose | None | list[A.BasicTransform] = None,
#         aug: AugmentationSequential = None,
#         sensors: Sequence[str] = ["S1", "S2"],
#         as_time_series: bool = False,
#         metadata_filename: str = default_metadata_filename,
#         max_cloud_percentage: float | None = None,
#         max_red_mean: float | None = None,
#         include_corrupt: bool = True,
#         **kwargs: Any,
#     ) -> None:
#         super().__init__(BioMasstersNonGeo, batch_size, num_workers, **kwargs)
#         self.data_root = data_root


#         self.bands = bands
#         self.train_transform = wrap_in_compose_is_list(train_transform)
#         self.val_transform = wrap_in_compose_is_list(val_transform)
#         self.test_transform = wrap_in_compose_is_list(test_transform)

#         self.sensors = sensors,
#         self.as_time_series = as_time_series
#         self.metadata_filename = metadata_filename
#         self.max_cloud_percentage = max_cloud_percentage
#         self.max_red_mean = max_red_mean
#         self.include_corrupt = include_corrupt

#     def setup(self, stage: str) -> None:
#         if stage in ["fit"]:
#             self.train_dataset = self.dataset_class(
#                 split="train",
#                 data_root=self.data_root,
#                 transform=self.train_transform,
#                 bands=self.bands,
#                 sensors=self.sensors,
#                 as_time_series=self.as_time_series,
#                 metadata_filename=self.metadata_filename,
#                 max_cloud_percentage=self.max_cloud_percentage,
#                 max_red_mean=self.max_red_mean,
#                 include_corrupt=self.include_corrupt,
#             )
#         if stage in ["fit", "validate"]:
#             self.val_dataset = self.dataset_class(
#                 split="val",
#                 data_root=self.data_root,
#                 transform=self.val_transform,
#                 bands=self.bands,
#                 sensors=self.sensors,
#                 as_time_series=self.as_time_series,
#                 metadata_filename=self.metadata_filename,
#                 max_cloud_percentage=self.max_cloud_percentage,
#                 max_red_mean=self.max_red_mean,
#                 include_corrupt=self.include_corrupt,
#             )
#         if stage in ["test"]:
#             self.test_dataset = self.dataset_class(
#                 split="val",
#                 data_root=self.data_root,
#                 transform=self.test_transform,
#                 bands=self.bands,
#                 sensors=self.sensors,
#                 as_time_series=self.as_time_series,
#                 metadata_filename=self.metadata_filename,
#                 max_cloud_percentage=self.max_cloud_percentage,
#                 max_red_mean=self.max_red_mean,
#                 include_corrupt=self.include_corrupt,
#             )
