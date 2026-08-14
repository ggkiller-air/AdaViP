# No-tactile DP sanity check

This experiment is independent of Table 2. Its only purpose is to overfit one
ALOHA-style episode and validate the DP checkpoint, inference, and deployment
interfaces before tactile observations are added.

The current protocol is:

- observations: `cam_high`, `cam_left_wrist`, `cam_right_wrist`, and 14D `qpos`;
- target: the recorded 14D `action`;
- source rate: 30 Hz, preserved without temporal downsampling;
- horizon: 16, observation steps: 2;
- no tactile keys, RDP stages, robot runner, ROS, or online rollout.

The source and converted data are at:

```text
/data/wangzihao/datasets/real_world/dp_sanity_no_tactile/
  raw/episode_0.hdf5
  processed_30hz/replay_buffer.zarr
```

The conversion is complete. For a future replacement or additional episodes:

```bash
/public/home/wangzihao/envs/rdp-baseline-py310/bin/python \
  scripts/real_world/convert_aloha_hdf5.py \
  /data/wangzihao/datasets/real_world/dp_sanity_no_tactile/raw/episode_0.hdf5 \
  --output /data/wangzihao/datasets/real_world/dp_sanity_no_tactile/processed_30hz \
  --overwrite
```

Run a bounded preflight inside an interactive GPU allocation:

```bash
DP_SANITY_NUM_EPOCHS=2 \
DP_SANITY_MAX_TRAIN_STEPS=5 \
DP_SANITY_WANDB_MODE=disabled \
bash scripts/real_world/train_dp_sanity.sh
```

After checking memory, utilization, workers, and checkpoint creation, submit the
overfit run:

```bash
DP_SANITY_WANDB_MODE=online sbatch slurm/real_world/train_dp_sanity.sbatch
```

Defaults are 200 epochs, batch size 32, four workers, and checkpoint/sample
logging every ten epochs. Outputs are written under
`/data/wangzihao/outputs/real_world/dp_sanity_no_tactile/`.

Run checkpoint inference as a short GPU batch job:

```bash
sbatch slurm/real_world/infer_dp_sanity.sbatch
```

The smoke test restores the EMA policy and normalizer, reads a two-frame window
from the converted episode, uses eight diffusion inference steps, and checks
the `[1, 15, 14]` rollout action. Its JSON report is written below the output
root in `inference/`. This is an offline interface check; robot I/O and online
deployment remain outside this repository.

Export a compact, deployment-neutral inference artifact on a CPU compute node:

```bash
sbatch slurm/real_world/export_dp_sanity_inference.sbatch
```

The artifact contains one EMA policy state dict (including the normalizer), the
resolved policy configuration, and a machine-readable observation/action
contract. It omits the raw model, optimizer, scheduler, and other training
resume state from the full workspace checkpoint. Validate the exported file by
passing it to the same GPU smoke:

```bash
sbatch slurm/real_world/infer_dp_sanity.sbatch \
  --checkpoint /data/wangzihao/outputs/real_world/dp_sanity_no_tactile/inference/dp_sanity_no_tactile_ema.pt
```
