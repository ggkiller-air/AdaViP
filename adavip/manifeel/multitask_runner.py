"""Task-wise ManiFeel evaluation runner for a 7D multi-task policy."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

import numpy as np
import torch

from adavip.manifeel.action_adapter import truncate_action_for_task
from adavip.manifeel.task_embeddings import load_task_embeddings, validate_task_embedding_dim
from adavip.manifeel.task_protocol import (
    DEFAULT_TASK_SPECS,
    ManiFeelTaskSpec,
    coerce_task_specs,
)
from adavip.manifeel.video_selection import retain_outcome_videos
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner


class _TaskConditionedPolicy:
    """Inject a fixed task embedding and adapt 7D policy actions for one env."""

    def __init__(
        self,
        policy,
        task_embedding: np.ndarray,
        action_dim: int,
        embedding_mode: str = "correct",
        alternate_task_embedding: np.ndarray | None = None,
        embedding_source_task_id: str | None = None,
        record_actions: bool = False,
    ) -> None:
        self.policy = policy
        embedding = np.asarray(task_embedding, dtype=np.float32)
        if embedding_mode == "correct":
            pass
        elif embedding_mode == "other_task":
            if alternate_task_embedding is None:
                raise ValueError("other_task mode requires an alternate task embedding")
            embedding = np.asarray(alternate_task_embedding, dtype=np.float32)
            if embedding.shape != np.asarray(task_embedding).shape:
                raise ValueError(
                    "Alternate task embedding shape mismatch: "
                    f"expected {np.asarray(task_embedding).shape}, got {embedding.shape}"
                )
        elif embedding_mode == "zero":
            embedding = np.zeros_like(embedding)
        else:
            raise ValueError(
                f"Unsupported MANIFEEL_TASK_EMBEDDING_MODE: {embedding_mode!r}"
            )
        self.task_embedding = torch.as_tensor(
            embedding, dtype=policy.dtype, device=policy.device
        )
        self.action_dim = action_dim
        self.embedding_mode = embedding_mode
        self.embedding_source_task_id = embedding_source_task_id or embedding_mode
        self.record_actions = record_actions
        self.action_chunks: list[np.ndarray] = []
        self.action_predictions: list[np.ndarray] = []

    @property
    def device(self):
        return self.policy.device

    @property
    def dtype(self):
        return self.policy.dtype

    def reset(self) -> None:
        self.policy.reset()

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        first = next(iter(obs_dict.values()))
        batch_size, n_obs_steps = first.shape[:2]
        embedding = self.task_embedding.view(1, 1, -1).expand(batch_size, n_obs_steps, -1)
        obs_dict = dict(obs_dict)
        obs_dict["task_embedding"] = embedding
        result = self.policy.predict_action(obs_dict)
        result["action"] = truncate_action_for_task(result["action"], self.action_dim)
        if self.record_actions:
            self.action_chunks.append(result["action"].detach().cpu().numpy())
            if "action_pred" in result:
                self.action_predictions.append(
                    result["action_pred"].detach().cpu().numpy()
                )
        return result

    def save_action_trajectory(self, path: Path) -> None:
        """Save executed chunks and full policy predictions for later comparison."""

        if not self.action_chunks:
            return
        chunks = np.stack(self.action_chunks, axis=0)
        payload: dict[str, np.ndarray] = {
            "action_chunks": chunks,
            "executed_actions": np.concatenate(self.action_chunks, axis=1),
            "task_embedding": self.task_embedding.detach().cpu().numpy(),
            "embedding_mode": np.asarray(self.embedding_mode),
            "embedding_source_task_id": np.asarray(self.embedding_source_task_id),
            "action_dim": np.asarray(self.action_dim, dtype=np.int64),
        }
        if self.action_predictions:
            payload["action_predictions"] = np.stack(
                self.action_predictions, axis=0
            )
        np.savez_compressed(path, **payload)


class MultiTaskManiFeelRunner(BaseImageRunner):
    """Run each ManiFeel task separately and log task-prefixed metrics."""

    def __init__(
        self,
        output_dir: str,
        shape_meta: dict,
        task_embedding_path: str,
        task_specs: list[dict] | None = None,
        n_test: int = 50,
        n_test_vis: int = 2,
        test_start_seed: int = 100000,
        max_steps: int = 500,
        n_obs_steps: int = 2,
        n_action_steps: int = 8,
        fps: int = 10,
        crf: int = 22,
        past_action: bool = False,
        tqdm_interval_sec: float = 5.0,
        tactile_size: list[int] | None = None,
    ) -> None:
        super().__init__(output_dir)
        self.shape_meta = shape_meta
        self.task_specs = coerce_task_specs(task_specs)
        embedding_specs = list(DEFAULT_TASK_SPECS)
        configured_ids = {spec.task_id for spec in embedding_specs}
        embedding_specs.extend(
            spec for spec in self.task_specs if spec.task_id not in configured_ids
        )
        self.task_embeddings = load_task_embeddings(task_embedding_path, embedding_specs)
        validate_task_embedding_dim(self.shape_meta, self.task_embeddings)
        self.n_test = n_test
        self.n_test_vis = n_test_vis
        self.test_start_seed = test_start_seed
        self.max_steps = max_steps
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.fps = fps
        self.crf = crf
        self.past_action = past_action
        self.tqdm_interval_sec = tqdm_interval_sec
        self.tactile_size = tactile_size or [224, 224]
        self.retain_outcome_videos = os.environ.get(
            "MANIFEEL_EVAL_RETAIN_OUTCOME_VIDEOS", "0"
        ) == "1"
        self.embedding_mode = os.environ.get(
            "MANIFEEL_TASK_EMBEDDING_MODE", "correct"
        )
        self.embedding_source_task_id = os.environ.get(
            "MANIFEEL_TASK_EMBEDDING_SOURCE_TASK_ID"
        )
        self.save_action_trajectories = os.environ.get(
            "MANIFEEL_EVAL_SAVE_ACTION_TRAJECTORIES", "0"
        ) == "1"

    def _shape_meta_for_task(self, spec: ManiFeelTaskSpec) -> dict:
        return {
            "obs": {
                key: value
                for key, value in self.shape_meta["obs"].items()
                if key != "task_embedding"
            },
            "action": {"shape": [spec.action_dim]},
        }

    def run(self, policy) -> dict:
        # Import lazily so CPU-side protocol tests can import this module without
        # touching IsaacGym. ManiFeel's train.py imports isaacgym before torch.
        from manifeel.env_runner.vistac_pih_runner import ManifeelRunner

        if len(self.task_specs) != 1:
            raise RuntimeError(
                "Multi-task evaluation requires one task per process; "
                "use scripts/manifeel/eval_multitask_dp.sh."
            )

        log_data: dict[str, float] = {}
        success_rates = []
        for task_index, spec in enumerate(self.task_specs):
            task_output_dir = Path(self.output_dir) / spec.task_id
            task_output_dir.mkdir(parents=True, exist_ok=True)
            runner = ManifeelRunner(
                output_dir=str(task_output_dir),
                shape_meta=self._shape_meta_for_task(spec),
                isaacgym_cfg_name=spec.isaacgym_cfg_name,
                n_test=self.n_test,
                n_test_vis=self.n_test if self.retain_outcome_videos else self.n_test_vis,
                test_start_seed=self.test_start_seed + task_index,
                max_steps=self.max_steps,
                n_obs_steps=self.n_obs_steps,
                n_action_steps=self.n_action_steps,
                fps=self.fps,
                crf=self.crf,
                past_action=self.past_action,
                tqdm_interval_sec=self.tqdm_interval_sec,
                tactile_size=self.tactile_size,
            )
            alternate_embedding = None
            embedding_source_task_id = spec.task_id
            if self.embedding_mode == "other_task":
                source_task_id = self.embedding_source_task_id
                if not source_task_id:
                    raise RuntimeError(
                        "MANIFEEL_TASK_EMBEDDING_SOURCE_TASK_ID is required for "
                        "other_task mode"
                    )
                source_specs = {
                    candidate.task_id: candidate for candidate in DEFAULT_TASK_SPECS
                }
                if source_task_id not in source_specs:
                    raise RuntimeError(f"Unknown embedding source task: {source_task_id}")
                if source_task_id == spec.task_id:
                    raise RuntimeError("Other-task embedding must use a different task")
                if source_specs[source_task_id].action_dim != spec.action_dim:
                    raise RuntimeError(
                        "Other-task embedding must match the evaluated action dimension: "
                        f"{source_task_id} has {source_specs[source_task_id].action_dim}D, "
                        f"{spec.task_id} has {spec.action_dim}D"
                    )
                alternate_embedding = self.task_embeddings[source_task_id]
                embedding_source_task_id = source_task_id
            elif self.embedding_mode == "zero":
                embedding_source_task_id = "zero"

            conditioned_policy = _TaskConditionedPolicy(
                policy=policy,
                task_embedding=self.task_embeddings[spec.task_id],
                action_dim=spec.action_dim,
                embedding_mode=self.embedding_mode,
                alternate_task_embedding=alternate_embedding,
                embedding_source_task_id=embedding_source_task_id,
                record_actions=self.save_action_trajectories,
            )
            try:
                task_log = runner.run(conditioned_policy)
            finally:
                if self.save_action_trajectories:
                    conditioned_policy.save_action_trajectory(
                        Path(self.output_dir) / "action_trajectory.npz"
                    )
                runner.env.close()
            if self.retain_outcome_videos:
                task_log = retain_outcome_videos(
                    task_log,
                    output_dir=Path(self.output_dir),
                    task_id=spec.task_id,
                )
            for key, value in task_log.items():
                if key.startswith("test/"):
                    clean_key = key[len("test/") :]
                    log_data[f"test/{spec.task_id}/{clean_key}"] = value
            success_key = f"test/{spec.task_id}/success_rate"
            if success_key in log_data and np.isfinite(log_data[success_key]):
                success_rates.append(log_data[success_key])

        if success_rates:
            log_data["test/macro_success_rate"] = float(np.mean(success_rates))
        return log_data
