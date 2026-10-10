"""Device-aware ManiFeel runner for physical multi-GPU hosts."""

from __future__ import annotations

import hydra
from omegaconf import OmegaConf

from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from manifeel.env_runner.vistac_pih_runner import ManifeelRunner
from manifeel.env_runner.vistac_pih_runner_unit import ManifeelRunner as UnitManifeelRunner
from manifeel.envs.vistac_isaacgym_multiple_env_wrapper import MultipleIsaacEnvWrapper
from manifeel.gym_util.video_recording_wrapper import VideoRecordingWrapper

from adavip.manifeel.full_info_wrapper import FullInfoMultiStepWrapper


class DeviceAwareManifeelRunner(ManifeelRunner):
    """Construct a ManiFeel task on one explicit physical CUDA device."""

    def __init__(
        self,
        output_dir: str,
        shape_meta: dict,
        isaacgym_cfg_name: str,
        device: str,
        n_test: int = 22,
        n_test_vis: int = 6,
        test_start_seed: int = 10000,
        max_steps: int = 200,
        n_obs_steps: int = 8,
        n_action_steps: int = 8,
        fps: int = 10,
        crf: int = 22,
        past_action: bool = False,
        tqdm_interval_sec: float = 5.0,
        tactile_size: list[int] | None = None,
    ) -> None:
        BaseImageRunner.__init__(self, output_dir)
        if not device.startswith("cuda:"):
            raise ValueError(f"device must be cuda:<physical-index>, got {device!r}")
        physical_device_id = int(device.split(":", 1)[1])

        isaacgym_cfg = hydra.compose(config_name=isaacgym_cfg_name)
        isaacgym_cfg.shape_meta = OmegaConf.create(shape_meta)
        isaacgym_cfg.sim_device = device
        isaacgym_cfg.rl_device = device
        isaacgym_cfg.graphics_device_id = physical_device_id
        if n_test is not None:
            isaacgym_cfg.num_envs = n_test

        steps_per_render = max(10 // fps, 1)
        self.env = FullInfoMultiStepWrapper(
            VideoRecordingWrapper(
                MultipleIsaacEnvWrapper(isaacgym_cfg),
                output_dir=output_dir,
                n_records=n_test_vis,
                fps=fps,
                crf=crf,
                file_paths=None,
                steps_per_render=steps_per_render,
            ),
            n_obs_steps=n_obs_steps,
            n_action_steps=n_action_steps,
            max_episode_steps=max_steps,
        )

        self.test_seed = test_start_seed
        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.past_action = past_action
        self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec
        self.tactile_size = tactile_size or [256, 256]


class DeviceAwareUnitManifeelRunner(DeviceAwareManifeelRunner):
    """Use physical GPU ordinals with the unit runner's tactile rollout."""

    def run(self, policy: BaseImagePolicy) -> dict:
        """Preserve the unit runner's rollout and success aggregation."""
        return UnitManifeelRunner.run(self, policy)
