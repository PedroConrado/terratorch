"""
This module handles registering multimae models into timm.
"""
import logging
import torch
import numpy as np
from functools import partial

from terratorch.datasets.utils import Modalities
from terratorch.models.backbones.multimae.multimae import MultiMAE, MultiViT
from terratorch.models.backbones.multimae.criterion import MaskedMSELoss, MaskedCrossEntropyLoss
from terratorch.models.backbones.multimae.input_adapters import PatchedInputAdapter, SemSegInputAdapter
from terratorch.models.backbones.multimae.output_adapters import SpatialOutputAdapter, ConvNeXtAdapter
from terratorch.registry import TERRATORCH_BACKBONE_REGISTRY

logger = logging.getLogger(__name__)


# TODO: make these user definable
DOMAIN_CONF = {
    Modalities.S1RTC: {
        "channels": 2,
        "stride_level": 1,
        "input_adapter": partial(PatchedInputAdapter, num_channels=2),
        "output_adapter": partial(SpatialOutputAdapter, num_channels=2),
        "loss": MaskedMSELoss,
        "image_size": 224,
        "patch_size": 16,
    },
    Modalities.S1GRD: {
        "channels": 2,
        "stride_level": 1,
        "input_adapter": partial(PatchedInputAdapter, num_channels=2),
        "output_adapter": partial(SpatialOutputAdapter, num_channels=2),
        "loss": MaskedMSELoss,
        "image_size": 224,
        "patch_size": 16,
    },
    Modalities.S2L1C: {
        "channels": 13,
        "stride_level": 1,
        "input_adapter": partial(PatchedInputAdapter, num_channels=13),
        "output_adapter": partial(SpatialOutputAdapter, num_channels=13),
        "loss": MaskedMSELoss,
        "image_size": 224,
        "patch_size": 16,
    },
    Modalities.S2L2A: {
        "channels": 12,
        "stride_level": 1,
        "input_adapter": partial(PatchedInputAdapter, num_channels=12),
        "output_adapter": partial(SpatialOutputAdapter, num_channels=12),
        "loss": MaskedMSELoss,
        "image_size": 224,
        "patch_size": 16,
    },
    Modalities.S2RGB: {
        "channels": 3,
        "stride_level": 1,
        "input_adapter": partial(PatchedInputAdapter, num_channels=3),
        "output_adapter": partial(SpatialOutputAdapter, num_channels=3),
        "loss": MaskedMSELoss,
        "image_size": 224,
        "patch_size": 16,
    },
    Modalities.DEM: {
        "channels": 1,
        "stride_level": 1,
        "input_adapter": partial(PatchedInputAdapter, num_channels=1),
        "output_adapter": partial(SpatialOutputAdapter, num_channels=1),
        "loss": MaskedMSELoss,
        "image_size": 224,
        "patch_size": 16,
    },
    Modalities.NDVI: {
        "channels": 1,
        "stride_level": 1,
        "input_adapter": partial(PatchedInputAdapter, num_channels=1),
        "output_adapter": partial(SpatialOutputAdapter, num_channels=1),
        "loss": MaskedMSELoss,
        "image_size": 224,
        "patch_size": 16,
    },
    Modalities.LULC: {
        "classes": 10,
        "stride_level": 1,
        "input_adapter": partial(SemSegInputAdapter, num_classes=10),
        "output_adapter": partial(SpatialOutputAdapter, num_channels=10),
        "loss": partial(MaskedCrossEntropyLoss, label_smoothing=0.0),
        "image_size": 224,
        "patch_size": 16,
    },
    'segmentation': {  # TODO: Test generalized semseg head from MultiMAE!
        "classes": 10,
        "stride_level": 1,
        "input_adapter": partial(SemSegInputAdapter, num_classes=10),
        "output_adapter": partial(SpatialOutputAdapter, num_channels=10),
        "loss": partial(MaskedCrossEntropyLoss, label_smoothing=0.0),
        "image_size": 224,
        "patch_size": 16,
    },
}


def _instantiate_input_adapter_from_dict(spec: dict) -> PatchedInputAdapter | SemSegInputAdapter:
    return spec["input_adapter"](
        stride_level=spec["stride_level"],
        patch_size_full=spec["patch_size"],
        image_size=spec["image_size"],
    )


def _parse_input_adapters(
    adapter_spec: list | dict[str, str | dict[str, int | str]],
) -> dict[str, PatchedInputAdapter | SemSegInputAdapter]:

    if isinstance(adapter_spec, list):
        # list to dict
        adapter_spec = {m: m for m in adapter_spec}
    if isinstance(adapter_spec, dict) and len(set(adapter_spec.keys())) != len(adapter_spec.keys()):
        msg = "Duplicate keys in input adapters"
        raise Exception(msg)
    input_adapters = {}

    for adapter_name, spec in adapter_spec.items():
        match spec:
            case str(spec):
                try:
                    spec = Modalities(spec.upper())
                except ValueError:
                    pass

                if spec in DOMAIN_CONF.keys():
                    input_adapters[adapter_name] = _instantiate_input_adapter_from_dict(DOMAIN_CONF[spec])
                else:
                    msg = f"Input Domain {adapter_name} does not exist. Choose one of {list(DOMAIN_CONF.keys())}"
                    raise ValueError(msg)
            case {"type": "PatchedInputAdapter", "num_channels": num_channels, **kwargs}:
                input_adapters[adapter_name] = PatchedInputAdapter(num_channels=num_channels, **kwargs)
            case {"type": "SemSegInputAdapter", "num_classes": num_classes,  **kwargs}:
                input_adapters[adapter_name] = SemSegInputAdapter(num_classes=num_classes, **kwargs)
            case _:
                msg = f"Invalid input adapter config for adapter {adapter_name}"
                raise ValueError(msg)
    return input_adapters


def _instantiate_output_adapter_from_dict(spec: dict, task: str, context_tasks: list[str], # num_channels: int
                                          ) -> SpatialOutputAdapter | ConvNeXtAdapter:
    return spec["output_adapter"](
        stride_level=spec["stride_level"],
        patch_size_full=spec["patch_size"],
        image_size=spec["image_size"],
        task=task,
        context_tasks=context_tasks,
        # num_channels=spec['channels'],  # TODO: Not passed in pretraining code
    )


def _instantiate_loss_from_dict(spec: dict) -> MaskedMSELoss | MaskedCrossEntropyLoss:
    return spec["loss"](
        patch_size=spec["patch_size"],
        stride=spec["stride_level"],
    )


def _parse_output_adapters(
    adapter_spec: list | dict[str, str | dict[str, int | str]],
) -> dict[str, SpatialOutputAdapter | SpatialOutputAdapter]:

    if isinstance(adapter_spec, list):
        # list to dict
        adapter_spec = {m: m for m in adapter_spec}
    if isinstance(adapter_spec, dict) and len(set(adapter_spec.keys())) != len(adapter_spec.keys()):
        msg = "Duplicate keys in output adapters"
        raise Exception(msg)
    output_adapters = {}
    loss_functions = {}

    for adapter_name, spec in adapter_spec.items():
        match spec:
            case str(spec):
                try:
                    spec = Modalities(spec.upper())
                except ValueError:
                    pass

                if spec in DOMAIN_CONF.keys():
                    output_adapters[adapter_name] = _instantiate_output_adapter_from_dict(
                        DOMAIN_CONF[spec],
                        task=adapter_name,
                        context_tasks=list(adapter_spec.keys()),
                    )
                    loss_functions[adapter_name] = _instantiate_loss_from_dict(DOMAIN_CONF[spec])
                else:
                    msg = f"output Domain {adapter_name} does not exist. Choose one of {list(DOMAIN_CONF.keys())}"
                    raise ValueError(msg)
            case {"type": "SpatialOutputAdapter", "num_channels": num_channels, "patch_size": patch_size, **kwargs}:
                output_adapters[adapter_name] = SpatialOutputAdapter(
                    task=adapter_name,
                    context_tasks=list(adapter_spec.keys()),
                    patch_size_full=patch_size,
                    num_channels=num_channels,
                    **kwargs
                )
                loss_functions[adapter_name] = MaskedMSELoss(patch_size=patch_size)
            case {"type": "ConvNeXtAdapter", "num_classes": num_classes,  "patch_size": patch_size, **kwargs}:
                output_adapters[adapter_name] = ConvNeXtAdapter(
                    num_classes=num_classes,
                    patch_size=patch_size,
                    **kwargs
                )
                loss_functions[adapter_name] = MaskedCrossEntropyLoss(patch_size=patch_size)
            case _:
                msg = f"Invalid output adapter config for adapter {adapter_name}"
                raise ValueError(msg)
    return output_adapters, loss_functions


# If you need to adapt the checkpoint file, do it here
def checkpoint_filter_fn(
    modalities: list[str],
    state_dict: dict[str, torch.Tensor],
    model: torch.nn.Module,
):
    new_state_dict = {}

    for k, v in state_dict.items():
        if "output_adapters" in k:
            continue

        # drop pos emb
        if "pos_emb" in k:
            continue

        if k.startswith("input_adapters."):
            try:
                modality_name = k.split(".")[1]
                modality = Modalities(modality_name)
            except ValueError:
                print(f"Modality {modality_name} is not in allowed modalities. Skipping {k}.")
                continue
            if modality.value not in modalities:
                print(f"Removing input adapter for {modality_name}: {k}")
                continue
        if k.startswith("encoder."):
            new_k = "layers." + k.removeprefix("encoder.")
            new_state_dict[new_k] = v
        else:
            new_state_dict[k] = v
    return new_state_dict


# class PrepareMultimodalFeaturesForDecoder:
#     def __init__(self, modalities: list[str], merging_method: str = 'concat'):
#         self.modalities = modalities
#         self.merging_method = merging_method
#
#     def __call__(self, x: list[torch.Tensor]) -> list[torch.Tensor]:
#         if len(x) == 2:
#             # MultiMAE decoder was used. Return predictions of first modality.
#             preds = list(x[0].values())
#             assert len(preds) != 1, "Terratorch can only handle one output modality."
#             return preds[0]
#
#         for output_index in range(len(x)):
#             x[output_index] = x[output_index].permute(0, 2, 1)
#             x[output_index] = x[output_index][:, :, :-1]  # remove global token
#             img_shape = int(np.sqrt(x[output_index].shape[-1] / len(self.modalities)))
#             if self.merging_method == 'concat':
#                 x[output_index] = x[output_index].reshape(x[output_index].shape[0], -1, img_shape, img_shape)
#             else:
#                 raise ValueError(f"Unsupported merging method {self.merging_method}")
#                 # TODO: Implement other methods, move to forward?
#
#         return x


def _create_multimae(
    variant: str,
    modalities: list[str],
    pretrained: bool = False,
    encoder_only: bool = True,
    merging_method: str = None,
    **kwargs,
):
    model_class = MultiViT if encoder_only else MultiMAE

    model = model_class(**kwargs)

    # TODO: load pre-trained

    return model


@TERRATORCH_BACKBONE_REGISTRY.register
def multimae_base(
    input_adapters: dict[str, str | dict[str, int | str]] | None = None,
    output_adapters: dict[str, str | dict[str, int | str]] | None = None,
    pretrained: bool = False,
    features_only: bool = True,
    **kwargs,
) -> torch.nn.Module:
    """MultiMAE base model."""

    if input_adapters is None:
        input_adapters = ['S1GRD', 'S1RTC', 'S2L1C', 'S2L2A', 'S2RGB', 'NDVI', 'DEM', 'LULC']
        logger.warning(f'Using default adapters.')
    input_adapters = _parse_input_adapters(input_adapters)

    if output_adapters is not None:
        output_adapters, loss_functions = _parse_output_adapters(output_adapters)

    model_args = {
        "input_adapters": input_adapters,
        "output_adapters": output_adapters,
        "loss_functions": loss_functions,
        "dim_tokens": 768,
        "depth": 12,
        "num_heads": 12,
        "mlp_ratio": 4,
        "qkv_bias": True,
        "norm_layer": partial(torch.nn.LayerNorm, eps=1e-6),
    }

    kwargs.pop('features_only', None)
    merging_method = None if output_adapters else kwargs.get('merging_method', 'concat')

    transformer = _create_multimae(
        "multimae_base",
        list(input_adapters.keys()),
        pretrained=pretrained,
        encoder_only=output_adapters is None,
        merging_method=merging_method,
        **dict(model_args, **kwargs),
    )
    return transformer


@TERRATORCH_BACKBONE_REGISTRY.register
def multimae_large(
    input_adapters: dict[str, str | dict[str, int | str]] | None = None,
    output_adapters: dict[str, str | dict[str, int | str]] | None = None,
    pretrained: bool = False,  # noqa: FBT002, FBT001
    **kwargs,
) -> torch.nn.Module:
    """MultiMAE large model."""

    if input_adapters is None:
        input_adapters = ['S1GRD', 'S1RTC', 'S2L1C', 'S2L2A', 'S2RGB', 'NDVI', 'DEM', 'LULC']
    input_adapters = _parse_input_adapters(input_adapters)

    if output_adapters is not None:
        output_adapters, loss_functions = _parse_output_adapters(output_adapters)

    model_args = {
        "input_adapters": input_adapters,
        "output_adapters": output_adapters,
        "loss_functions": loss_functions,
        "dim_tokens": 1024,
        "depth": 24,
        "num_heads": 16,
        "mlp_ratio": 4,
        "qkv_bias": True,
        "norm_layer": partial(torch.nn.LayerNorm, eps=1e-6),
    }

    kwargs.pop('features_only', None)
    merging_method = None if output_adapters else kwargs.get('merging_method', 'concat')

    transformer = _create_multimae(
        "multimae_large",
        list(input_adapters.keys()),
        pretrained=pretrained,
        encoder_only=output_adapters is None,
        merging_method=merging_method,
        **dict(model_args, **kwargs),
    )
    return transformer
