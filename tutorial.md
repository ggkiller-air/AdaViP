# Piper Pick-Cup 左臂训练与使用

在仓库根目录依次执行下面的命令。训练使用左臂 7 维状态和动作，以及
`cam_high`、`cam_left_wrist` 两路相机；默认训练 200 epochs。

```bash
cd /public/home/wangzihao/projects/AdaViP
set -euo pipefail

/public/home/wangzihao/envs/rdp-baseline-py310/bin/hf auth login

HF_HUB_DISABLE_XET=1 \
HF_HUB_DOWNLOAD_TIMEOUT=600 \
HF_HUB_ETAG_TIMEOUT=60 \
/public/home/wangzihao/envs/rdp-baseline-py310/bin/hf download \
  ggkiller-air/piper-pick-cup \
  --repo-type dataset \
  --revision 5be8aaf9d353b2d7f199cadeba5dd2eff92c454b \
  --local-dir /data/wangzihao/datasets/real_world/piper-pick-cup \
  --max-workers 1

/public/home/wangzihao/envs/rdp-baseline-py310/bin/python \
  scripts/real_world/convert_aloha_hdf5.py \
  /data/wangzihao/datasets/real_world/piper-pick-cup/pick_cup/episode_0.hdf5 \
  --output /data/wangzihao/datasets/real_world/piper-pick-cup/processed_30hz \
  --overwrite

DP_SANITY_DATASET_PATH=/data/wangzihao/datasets/real_world/piper-pick-cup/processed_30hz \
DP_SANITY_OUTPUT_DIR=/data/wangzihao/outputs/real_world/piper-pick-cup-left/seed_42 \
DP_SANITY_TASK_CONFIG=piper_pick_cup_left_30hz \
DP_SANITY_WANDB_MODE=disabled \
sbatch --wait slurm/real_world/train_dp_sanity.sbatch

source scripts/real_world/common.sh
activate_rdp
python scripts/real_world/export_dp_sanity_inference.py \
  --checkpoint /data/wangzihao/outputs/real_world/piper-pick-cup-left/seed_42/checkpoints/latest.ckpt \
  --output /data/wangzihao/outputs/real_world/piper-pick-cup-left/inference/piper_pick_cup_left_ema.pt \
  --control-frequency-hz 30 \
  --action-semantics \
  'left arm absolute joint-position targets: joints 0-5 and gripper 6; radians/metres as recorded' \
  --overwrite

export NO_PROXY="${NO_PROXY:+${NO_PROXY},}.xethub.hf.co"
export no_proxy="${NO_PROXY}"
export HF_XET_HIGH_PERFORMANCE=1
export HF_HUB_ETAG_TIMEOUT=60
python scripts/real_world/upload_piper_pick_cup_ema.py
```

最后一个命令会创建公开模型仓库 `ggkiller-air/piper-pick-cup-dp`，并在同一个
commit 中上传 EMA artifact、`README.md` 和 `artifact_manifest.json`。

## 下载与推理

```bash
/public/home/wangzihao/envs/rdp-baseline-py310/bin/hf download \
  ggkiller-air/piper-pick-cup-dp \
  piper_pick_cup_left_ema.pt \
  artifact_manifest.json \
  --local-dir /data/wangzihao/checkpoints/real_world/piper-pick-cup-dp
```

输入为两帧 `cam_high [B,2,3,480,640]`、
`cam_left_wrist [B,2,3,480,640]` 和 `qpos [B,2,7]`，输出 action shape 为
`[B,15,7]`。

```python
import dill
import hydra
import torch
from omegaconf import OmegaConf

path = (
    "/data/wangzihao/checkpoints/real_world/piper-pick-cup-dp/"
    "piper_pick_cup_left_ema.pt"
)
with open(path, "rb") as stream:
    artifact = torch.load(stream, map_location="cpu", pickle_module=dill)

policy = hydra.utils.instantiate(OmegaConf.create(artifact["policy_config"]))
policy.load_state_dict(artifact["policy_state_dict"], strict=True)
policy.num_inference_steps = artifact["runtime"]["num_inference_steps"]
policy.eval().to("cuda:0")

obs = {key: value.to("cuda:0") for key, value in obs.items()}
with torch.inference_mode():
    action = policy.predict_action(obs)["action"]
```

真机执行层需要另外处理机器人通信、时间同步、限位和急停。
