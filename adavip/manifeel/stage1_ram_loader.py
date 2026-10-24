"""Opt-in lossless batching directly into shared RAM for Stage 1 v2."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset, default_collate, get_worker_info

from adavip.manifeel.full_sensor_schema import IMAGE_KEYS
from adavip.manifeel.stage1_parallel_data import CachedStage1Dataset
from adavip.manifeel.stage1_v2_schema import FUTURE_STEPS, OBS_HISTORY_STEPS


def shared_image_buffer(shape: tuple[int, ...]) -> torch.Tensor:
    """Allocate the final collated storage once, sharing it only inside workers."""
    prototype = torch.empty(0, dtype=torch.float32)
    if get_worker_info() is None:
        return prototype.new_empty(shape)
    storage = prototype._typed_storage()._new_shared(math.prod(shape), device=prototype.device)
    return prototype.new(storage).resize_(*shape)


def already_batched(batch: dict[str, Any]) -> dict[str, Any]:
    """Keep prebatched tensors intact instead of stacking the image buffers again."""
    if not isinstance(batch, dict):
        raise TypeError("RAM loader requires a prebatched Stage 1 dataset")
    return batch


class RamBatchDataset(Dataset):
    """Preserve sample order and metadata while copying each image only once."""

    def __init__(self, dataset: CachedStage1Dataset) -> None:
        if type(dataset) is not CachedStage1Dataset:
            raise TypeError("RAM batching requires CachedStage1Dataset")
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.dataset[index]

    def __getitems__(self, indices: list[int]) -> dict[str, Any]:
        if not indices:
            raise ValueError("Cannot assemble an empty batch")
        images = {
            name: {
                key: shared_image_buffer((len(indices), steps, *self.dataset.shape_meta["obs"][key]["shape"]))
                for key in IMAGE_KEYS
            }
            for name, steps in (("obs", OBS_HISTORY_STEPS), ("future_obs", FUTURE_STEPS))
        }
        samples = []
        for position, index in enumerate(indices):
            destinations = {name: {key: values[position] for key, values in sensors.items()}
                            for name, sensors in images.items()}
            sample = self.dataset._get_sample(index, destinations)
            for name in images:
                for key in IMAGE_KEYS:
                    del sample[name][key]
            samples.append(sample)
        result = default_collate(samples)
        for name, sensors in images.items():
            result[name] = {**sensors, **result[name]}
        return result


def make_ram_dataloader(dataset: CachedStage1Dataset, **options: Any) -> DataLoader:
    """Use original sampler, batch size, pinning and prefetch settings unchanged."""
    if "collate_fn" in options:
        raise ValueError("RAM batching owns collate_fn")
    if options.get("batch_size", 1) is None and options.get("batch_sampler") is None:
        raise ValueError("RAM loader requires automatic batching")
    return DataLoader(RamBatchDataset(dataset), collate_fn=already_batched, **options)
