"""Lossless reads from the already verified, shared decoded sensor cache."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from adavip.manifeel.decoded_sensor_cache import SourceIndexedArray
from adavip.manifeel.stage1_v2_dataset import Stage1V2ManiFeelDataset


class CachedStage1Dataset(Stage1V2ManiFeelDataset):
    """Avoid repeated validation scans and intermediate copies of verified frames."""

    def _read_images(self, array: Any, indices: np.ndarray, key: str) -> np.ndarray:
        if not isinstance(array, SourceIndexedArray):
            return super()._read_images(array, indices, key)
        return np.ascontiguousarray(self._cached_image_view(array, indices, key))

    def _cached_image_view(self, array: SourceIndexedArray, indices: np.ndarray, key: str) -> np.ndarray:
        mapped = array.row_map[indices]
        if np.any(mapped < 0):
            raise IndexError("Verified cache does not contain the requested frame")
        if len(mapped) and np.all(np.diff(mapped) == 1):
            images = array.values[int(mapped[0]):int(mapped[-1]) + 1]
        else:
            images = array.values[mapped]
        expected = tuple(self.shape_meta["obs"][key]["shape"])
        if images.dtype != np.float32 or images.shape[1:] != (*expected[1:], expected[0]):
            raise ValueError(f"Verified cache shape/dtype mismatch for {key}")
        return np.moveaxis(images, -1, 1)

    def _image_tensor(
        self, array: Any, indices: np.ndarray, key: str, destination: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if destination is None or not isinstance(array, SourceIndexedArray):
            return super()._image_tensor(array, indices, key, destination)
        np.copyto(destination.numpy(), self._cached_image_view(array, indices, key), casting="no")
        return destination
