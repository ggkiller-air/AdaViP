"""Rollout adapter providing the same delta-frame tactile pairs as training."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np

from diffusion_policy.env_runner.base_image_runner import BaseImageRunner


class FilmFusionObservationAdapter:
    """Convert raw observation history into current/past tactile pairs."""

    def __init__(
        self,
        env: Any,
        n_obs_steps: int = 2,
        tactile_frame_delta: int = 5,
        tactile_source_key: str = "right_tactile_camera_taxim",
        tactile_pair_key: str = "right_tactile_camera_taxim_pair",
    ) -> None:
        if n_obs_steps < 1 or tactile_frame_delta < 0:
            raise ValueError("invalid observation history settings")
        self.env = env
        self.n_obs_steps = int(n_obs_steps)
        self.tactile_frame_delta = int(tactile_frame_delta)
        self.tactile_source_key = tactile_source_key
        self.tactile_pair_key = tactile_pair_key
        self.history_steps = self.n_obs_steps + self.tactile_frame_delta

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)

    def _adapt(self, observations: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        if self.tactile_source_key not in observations:
            raise KeyError(f"rollout observation lacks {self.tactile_source_key}")
        tactile = observations[self.tactile_source_key]
        if tactile.ndim != 5 or tactile.shape[1] != self.history_steps:
            raise ValueError(
                f"raw tactile history must be [B,{self.history_steps},C,H,W], "
                f"got {tactile.shape}"
            )
        current = tactile[:, self.tactile_frame_delta :]
        past = tactile[:, : self.n_obs_steps]
        result = {
            key: value[:, -self.n_obs_steps :]
            for key, value in observations.items()
            if key != self.tactile_source_key
        }
        result[self.tactile_pair_key] = np.concatenate((current, past), axis=2)
        return result

    def reset(self) -> dict[str, np.ndarray]:
        """Reset and adapt the padded initial history."""
        return self._adapt(self.env.reset())

    def step(self, action: np.ndarray):
        """Step the wrapped multi-step environment and adapt its history."""
        observations, reward, done, info = self.env.step(action)
        return self._adapt(observations), reward, done, info


class FilmFusionRunner(BaseImageRunner):
    """Compose the upstream ManiFeel runner with delta-frame observation logic."""

    def __init__(
        self,
        output_dir: str,
        shape_meta: dict[str, Any],
        isaacgym_cfg_name: str,
        n_test: int = 22,
        n_test_vis: int = 6,
        test_start_seed: int = 10000,
        max_steps: int = 200,
        n_obs_steps: int = 2,
        n_action_steps: int = 8,
        fps: int = 10,
        crf: int = 22,
        past_action: bool = False,
        tqdm_interval_sec: float = 5.0,
        tactile_frame_delta: int = 5,
        tactile_source_key: str = "right_tactile_camera_taxim",
        tactile_pair_key: str = "right_tactile_camera_taxim_pair",
        tactile_size: tuple[int, int] = (320, 240),
    ) -> None:
        super().__init__(output_dir)
        raw_shape_meta = copy.deepcopy(shape_meta)
        raw_shape_meta["obs"].pop(tactile_pair_key)
        raw_shape_meta["obs"][tactile_source_key] = {
            "shape": [3, *tactile_size],
            "type": "rgb",
        }

        # Delay this import so CPU dataset/model tests do not import Isaac Gym.
        from manifeel.env_runner.vistac_pih_runner_unit import ManifeelRunner

        history_steps = int(n_obs_steps) + int(tactile_frame_delta)
        runner = ManifeelRunner(
            output_dir=output_dir,
            shape_meta=raw_shape_meta,
            isaacgym_cfg_name=isaacgym_cfg_name,
            n_test=n_test,
            n_test_vis=n_test_vis,
            test_start_seed=test_start_seed,
            max_steps=max_steps,
            n_obs_steps=history_steps,
            n_action_steps=n_action_steps,
            fps=fps,
            crf=crf,
            past_action=past_action,
            tqdm_interval_sec=tqdm_interval_sec,
            tactile_size=list(tactile_size),
        )
        runner.env = FilmFusionObservationAdapter(
            runner.env,
            n_obs_steps=n_obs_steps,
            tactile_frame_delta=tactile_frame_delta,
            tactile_source_key=tactile_source_key,
            tactile_pair_key=tactile_pair_key,
        )
        runner.n_obs_steps = int(n_obs_steps)
        self._runner = runner
        self.env = runner.env

    def run(self, policy):
        """Delegate rollout and metric collection to the upstream runner."""
        return self._runner.run(policy)
