# Piper Wipe-Dish Baseline Protocol

This document is the shared data and evaluation contract for the real-robot
single-task DP and RDP baselines. It is intentionally separate from either
training implementation. Both implementations must follow it without
modifying the immutable source dataset.

## Scope

- Task: `wipe-dish`, single task, 45 demonstrations.
- Source: `/data/wangzihao/wipe-dish-rdp-both`.
- Source format: `piper_both_rdp_v2`, RGB color order, 30 Hz.
- Source replay buffer: 43,019 frames and 45 episode boundaries.
- Source image shape: HWC `[240, 320, 3]`, `uint8`.
- The source directory and its `replay_buffer.zarr` are read-only inputs. All
  derived arrays and reports belong in a separate prepared-data directory.

The first baseline excludes `robot_qvel`, timestamps, and any task-specific
labels. They may be added only as separately named ablations.

## Action and State Contract

The target is the recorded absolute joint action, with shape `[14]` and this
fixed order:

```text
[left_joint_1, left_joint_2, left_joint_3, left_joint_4,
 left_joint_5, left_joint_6, left_gripper,
 right_joint_1, right_joint_2, right_joint_3, right_joint_4,
 right_joint_5, right_joint_6, right_gripper]
```

The proprioceptive input is `robot_qpos` in the same order and shape. Do not
convert to TCP pose, reorder the arms, or change to delta/relative actions for
the baseline. The dataset records joint-space absolute targets, and both arms
have substantial motion.

## Episode Split

Split by complete episode, never by individual frames. The canonical split is
generated with `numpy.random.default_rng(seed=42)` and `val_ratio=0.20`, which
selects these nine validation episodes:

```text
[3, 9, 17, 25, 29, 30, 36, 40, 42]
```

The remaining 36 episodes are training episodes:

```text
[0, 1, 2, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 18,
 19, 20, 21, 22, 23, 24, 26, 27, 28, 31, 32, 33, 34, 35, 37,
 38, 39, 41, 43, 44]
```

Sequence windows must not cross episode boundaries. Edge padding, when needed
by a model window, repeats the edge frame within that episode.

## Sampling and Normalization

- Preserve the native 30 Hz frame stream in the prepared data.
- Use a one-second action chunk for the first comparison: `horizon=30` and
  `n_obs_steps=2`. Any later horizon or observation-rate change is a named
  ablation, not a silent configuration change.
- Convert RGB arrays from HWC `uint8` to CHW `float32` using `/255.0`.
  Keep RGB channel order. Model-side ImageNet normalization and random crops
  are configuration choices and must be recorded in each run.
- Fit one independent limits normalizer per low-dimensional key and per action
  key using training episodes only. Map each dimension to `[-1, 1]` from its
  training min/max; constant dimensions use the normalizer's epsilon rule.
- Use the same fitted normalizer artifact for all modality variants of a given
  method. Never fit statistics on validation episodes or on padded samples.
- Keep action normalization separate from qpos normalization even though the
  arrays are numerically similar.

## Input Variants

The visual camera set is fixed to `cam_high`, `cam_left_wrist`, and
`cam_right_wrist`. Tactile variants are additive to this same visual set.

### DP

Use the standard image-conditioned diffusion policy: one ResNet18 image branch
per RGB key (`share_rgb_model=false`), a low-dimensional projection for qpos
and any PCA keys, followed by the conditional 1D diffusion U-Net.
For the 30-frame horizon, the standard three-level temporal U-Net is retained.
It pads the temporal tensor to 32 steps internally for the two stride-2
downsampling stages, then crops the prediction back to the public 30-step
action horizon. This does not change the dataset or deployment action rate.

Run these variants with identical optimizer, horizon, split, and seed:

1. `dp_v`: three visual cameras + 14D qpos.
2. `dp_pca`: `dp_v` + left and right 15D GelSight embeddings.
3. `dp_rgb`: `dp_v` + raw left and right GelSight RGB images.

Do not feed raw RGB and PCA for the same GelSight stream in one main run.

### RDP

Use the existing two-stage structure without changing the pinned upstream
source:

- AT tokenizes 14D action chunks. The canonical tactile-conditioned variant
  uses full-rate PCA features as the AT temporal condition.
- LDP receives the three visual cameras, 14D qpos, and the selected tactile
  representation. Its RGB keys use the same ResNet18/ImageNet preprocessing as
  DP; PCA keys use low-dimensional projections.

The first RDP run is `rdp_pca`, matching the repository's existing Piper RDP
pattern. `rdp_rgb` is an explicitly named ablation: raw GelSight images enter
the LDP RGB encoder. If its AT does not have a learned raw-image temporal
encoder, keep AT action-only and record that distinction; do not claim it is
architecturally identical to `rdp_pca`.

## GelSight PCA Preparation

For each GelSight stream independently:

1. Detect and track the fixed 7x9 (63-marker) grid, using the first frame of
   each episode as the reference.
2. Normalize marker offsets by image width and height.
3. Fit a 15-component PCA on training frames only.
4. Store the resulting float32 embedding and the PCA mean/transform matrices
   in the prepared-data artifact, together with a detection audit.

The PCA detector must achieve complete frame coverage before `dp_pca` or
`rdp_pca` is accepted. A failed detector must not be silently replaced with
zeros or raw pixels.

## TacFF Status

TacFF is not part of this baseline matrix. The downloaded dataset contains
GelSight RGB images but no calibrated force-field tensor. The available TacFF
encoder expects a simulation-style force-field grid with defined units and
coordinate conventions; RGB cannot be passed to it as a substitute.

Re-opening TacFF requires a separately verified GelSight-to-force-field
calibration/solver and a new observation schema. Until then, report TacFF as
`not_available`, not as a failed model run.

## Parallel Ownership and Reproducibility

- The DP and RDP owners consume the same prepared-data artifact and the same
  split/normalizer manifests.
- DP outputs go under
  `/data/wangzihao/outputs/real_world/wipe_dish/dp/`; RDP outputs go under
  `/data/wangzihao/outputs/real_world/wipe_dish/rdp/`.
- Every run records method, modality variant, seed, git revision, source and
  prepared-data paths, split seed, horizon, observation keys, action order,
  and normalizer checksum.
- The first comparison should use `dp_v` and `rdp_pca`; the other variants are
  ablations after both canonical pipelines complete.

## DP Launcher

The dedicated DP entrypoint selects the prepared data and matching task config
from one modality variable. A short Slurm preflight can be run before formal
training:

```bash
WIPE_DISH_DP_MODALITY=pca \
WIPE_DISH_DP_WANDB_MODE=disabled \
WIPE_DISH_DP_NUM_EPOCHS=2 \
WIPE_DISH_DP_MAX_TRAIN_STEPS=5 \
bash scripts/real_world/train_wipe_dish_dp.sh
```

Use `WIPE_DISH_DP_MODALITY=rgb` for the raw GelSight image variant. Formal
training uses the Slurm entrypoint after checking GPU memory, data-loader
workers, and checkpoint behavior in the preflight allocation.
