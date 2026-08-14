"""Utilities for frozen task text embeddings used by ManiFeel protocols."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from adavip.manifeel.task_protocol import ManiFeelTaskSpec


def load_task_embeddings(
    path: str | Path,
    task_specs: list[ManiFeelTaskSpec],
) -> dict[str, np.ndarray]:
    """Load a task_id -> embedding mapping from a NumPy archive."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Task embedding archive not found: {path}. "
            "Generate it with scripts/manifeel/generate_task_embeddings.py."
        )

    archive = np.load(path, allow_pickle=False)
    if "task_ids" not in archive or "embeddings" not in archive:
        raise ValueError(f"Task embedding archive missing task_ids/embeddings: {path}")

    task_ids = [str(x) for x in archive["task_ids"].tolist()]
    embeddings = np.asarray(archive["embeddings"], dtype=np.float32)
    if embeddings.ndim != 2 or len(task_ids) != embeddings.shape[0]:
        raise ValueError(
            f"Invalid task embedding archive shapes: task_ids={len(task_ids)}, "
            f"embeddings={embeddings.shape}"
        )

    by_id = {task_id: embeddings[idx] for idx, task_id in enumerate(task_ids)}
    missing = [spec.task_id for spec in task_specs if spec.task_id not in by_id]
    if missing:
        raise KeyError(f"Missing task embeddings for task ids: {missing}")
    return {spec.task_id: by_id[spec.task_id] for spec in task_specs}


def embedding_dim(embeddings: dict[str, np.ndarray]) -> int:
    """Return the shared embedding dimension, validating consistency."""
    dims = {value.shape[-1] for value in embeddings.values()}
    if len(dims) != 1:
        raise ValueError(f"Task embeddings must share one dimension, got {sorted(dims)}")
    return next(iter(dims))


def expected_embedding_dim(shape_meta: dict) -> int:
    """Return the task embedding dimension declared by a ManiFeel shape_meta."""
    obs_meta = shape_meta.get("obs", {})
    if "task_embedding" not in obs_meta:
        raise KeyError("shape_meta['obs'] must include task_embedding for multi-task DP")
    shape = obs_meta["task_embedding"].get("shape")
    if not shape or len(shape) != 1:
        raise ValueError(f"task_embedding shape must be one-dimensional, got {shape}")
    return int(shape[0])


def validate_task_embedding_dim(shape_meta: dict, embeddings: dict[str, np.ndarray]) -> int:
    """Validate cached task embeddings against the configured model input shape."""
    expected_dim = expected_embedding_dim(shape_meta)
    actual_dim = embedding_dim(embeddings)
    if actual_dim != expected_dim:
        raise ValueError(
            f"Task embedding dim mismatch: config expects {expected_dim}, "
            f"archive provides {actual_dim}"
        )
    return actual_dim
