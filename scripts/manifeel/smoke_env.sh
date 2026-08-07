#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"
activate_manifeel
print_manifeel_context

if ! "${MANIFEEL_PYTHON}" -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable"'; then
    echo "GPU is required for the official Isaac Gym smoke test." >&2
    exit 2
fi

export HYDRA_FULL_ERROR=1
cd "${MANIFEEL_ROOT}"
"${MANIFEEL_PYTHON}" - <<'PY'
import hydra
import numpy as np
import os
from omegaconf import OmegaConf

from manifeel.envs.vistac_isaacgym_multiple_env_wrapper import MultipleIsaacEnvWrapper

OmegaConf.register_new_resolver("eval", eval, replace=True)
config_dir = os.path.join(os.environ["MANIFEEL_ROOT"], "manifeel", "config")
hydra.initialize_config_dir(config_dir=config_dir, version_base=None)
cfg = hydra.compose(config_name="isaacgym_config_usb.yaml")
cfg.num_envs = 1
cfg.headless = True
cfg.force_render = False
cfg.capture_video = False
cfg.shape_meta = OmegaConf.create({
    "obs": {"wrist": {"shape": [3, 256, 256], "type": "rgb"},
            "state": {"shape": [7], "type": "low_dim"}},
    "action": {"shape": [6]},
})

env = MultipleIsaacEnvWrapper(cfg)
obs = env.reset()
print("observation_keys", sorted(obs))
for key, value in sorted(obs.items()):
    print("observation", key, value.shape, value.dtype, value.device if hasattr(value, "device") else "cpu")
print("action_shape", env.action_space.shape)
action = np.zeros((env.num_envs, env.action_space.shape[0]), dtype=np.float32)
for step in range(5):
    obs, reward, reset, info = env.step(action)
    print("step", step, "reset_shape", reset.shape, "reward_shape", np.asarray(reward).shape)
print("official headless environment reset/step: OK")
PY
