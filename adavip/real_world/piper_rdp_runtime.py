"""Deployment-side loading and online GelSight preprocessing for Piper RDP."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

from adavip.real_world.gelsight_marker_processor import (
    MarkerDetectorConfig,
    assign_to_reference,
    detect_markers,
    order_reference_grid,
)


ARTIFACT_FORMAT = "adavip.piper_rdp.inference.v1"


class OnlineGelSightPcaProcessor:
    """Track the 63-marker grid and apply the frozen task-local PCA online."""

    def __init__(
        self,
        transform_matrix: np.ndarray,
        mean: np.ndarray,
        detector_config: MarkerDetectorConfig = MarkerDetectorConfig(),
    ) -> None:
        self.transform_matrix = np.asarray(transform_matrix, dtype=np.float32)
        self.mean = np.asarray(mean, dtype=np.float32)
        expected_input = detector_config.marker_count * 2
        if self.transform_matrix.shape[0] != expected_input:
            raise ValueError(
                f"PCA transform expects {self.transform_matrix.shape[0]} inputs, "
                f"not {expected_input}"
            )
        if self.mean.shape != (expected_input,):
            raise ValueError(f"Unexpected PCA mean shape: {self.mean.shape}")
        self.detector_config = detector_config
        self.reference_xy: np.ndarray | None = None
        self.scale_xy: np.ndarray | None = None

    @classmethod
    def from_directory(cls, bundle_dir: Path) -> "OnlineGelSightPcaProcessor":
        """Load PCA files from an exported deployment directory."""
        pca_dir = bundle_dir / "gelsight_pca"
        return cls(
            transform_matrix=np.load(pca_dir / "pca_transform_matrix.npy"),
            mean=np.load(pca_dir / "pca_mean_matrix.npy"),
        )

    def reset(self) -> None:
        """Forget the current episode reference frame."""
        self.reference_xy = None
        self.scale_xy = None

    def initialize(self, no_contact_rgb: np.ndarray) -> dict[str, Any]:
        """Set the episode's no-contact reference from its first RGB frame."""
        detection = detect_markers(no_contact_rgb, self.detector_config)
        if not detection.success:
            raise RuntimeError(
                f"Reference detected {detection.component_count}/"
                f"{self.detector_config.marker_count} markers"
            )
        self.reference_xy = order_reference_grid(
            detection.centers_xy, self.detector_config
        )
        height, width = no_contact_rgb.shape[:2]
        self.scale_xy = np.asarray([width, height], dtype=np.float32)
        result = self.process(no_contact_rgb)
        result["reference_initialized"] = True
        return result

    def process(self, rgb: np.ndarray) -> dict[str, Any]:
        """Return normalized 126D marker offsets and their frozen 15D PCA."""
        if self.reference_xy is None or self.scale_xy is None:
            raise RuntimeError("Call initialize() with a no-contact frame first")
        detection = detect_markers(rgb, self.detector_config)
        if not detection.success:
            raise RuntimeError(
                f"Frame detected {detection.component_count}/"
                f"{self.detector_config.marker_count} markers"
            )
        tracked_xy, assignment_distance = assign_to_reference(
            self.reference_xy,
            detection.centers_xy,
            self.detector_config.max_assignment_distance_px,
        )
        offset = (tracked_xy - self.reference_xy) / self.scale_xy
        flattened = offset.reshape(-1)
        embedding = (flattened - self.mean) @ self.transform_matrix
        return {
            "initial_marker": (self.reference_xy / self.scale_xy).astype(np.float32),
            "marker_offset": offset.astype(np.float32),
            "marker_offset_emb": embedding.astype(np.float32),
            "detected_markers": detection.component_count,
            "threshold": detection.threshold,
            "assignment_mean_px": float(assignment_distance.mean()),
            "assignment_max_px": float(assignment_distance.max()),
            "detector": asdict(self.detector_config),
        }


def load_policy_artifact(
    artifact_path: Path, device: str | torch.device = "cpu"
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Instantiate and restore the self-contained EMA policy artifact."""
    target_device = torch.device(device)
    payload = torch.load(artifact_path, map_location="cpu", weights_only=False)
    if payload.get("format") != ARTIFACT_FORMAT:
        raise ValueError(f"Unsupported artifact format: {payload.get('format')!r}")
    policy_config = OmegaConf.create(payload["policy_config"])
    policy = hydra.utils.instantiate(policy_config)
    policy.load_state_dict(payload["policy_state_dict"], strict=True)
    # Upstream creates the EMA copy before set_normalizer(), so its nested AT
    # normalizer is empty even though the identical policy normalizer is saved.
    # Restore the training-time policy.set_normalizer() propagation explicitly.
    policy.at.set_normalizer(policy.normalizer)
    policy.num_inference_steps = int(payload["runtime"]["num_inference_steps"])
    policy.eval().to(target_device)
    metadata = {
        key: value for key, value in payload.items() if key != "policy_state_dict"
    }
    metadata["at_normalizer_synchronized_from_policy"] = True
    return policy, metadata
