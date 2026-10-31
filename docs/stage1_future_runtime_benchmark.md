# H200 runtime experience for Stage 1 future prediction

This independent, low-priority queue is for later experience. It does not resume,
pause, reconfigure or write checkpoints into any formal B--E training run. Formal
loss weights remain B0.25332773311046375, C0.019863649098324596,
D0.17650865668093083, E0.03004826433718963.

The benchmark uses B (raw tactile prediction with CNN decoders) and E (visual plus
tactile latent prediction with target EMA) as representative workloads. All cases
start from the identical original random Stage-1 initialization and fresh AdamW,
use batch32, the production loss weights and LR schedule, and run the original
policy EMA and future target EMA updates. Full decoded299GB arrays are loaded
and SHA256-verified once into shared RAM, with8persistent pinned workers and
nonblocking H2D. Data/noise streams match within each repeat across all profiles.

Profiles:

| Profile | Difference from production FP32 + cuDNN TF32 |
| --- | --- |
| baseline | None |
| bf16 | BF16 autocast; FP32 loss reductions, optimizer and EMA |
| channels_last | Online/target image CNNs and raw CNN decoders |
| cudnn_autotune | cuDNN benchmark enabled; determinism retained |
| matmul_tf32 | TensorFloat32 enabled for matrix multiplication too |
| fused_adamw | Same AdamW hyperparameters with fused CUDA implementation |
| compile_unet | Compile only the U-Net forward; preserve module tree for EMA |
| bf16_channels_last | Combine BF16 and image memory layout |
| bf16_channels_last_fused_compile | Combine BF16/layout/fused AdamW/compiled U-Net |

Each profile runs24warmup updates plus120timed updates, twice, with randomized
case ordering:36cases in total. Timed work includes DataLoader waits, H2D,
forward/backward, gradient clipping, optimizer and both EMA updates. Compile
and first-use costs are recorded in warmup time separately from steady throughput.
First-step Action/Future/Total Loss, per-ResNet Future gradients, fusion-gradient
norms/ratios/cosines, comparison against the corresponding FP32 baseline, peak
allocator memory and actual worker affinities are saved. FP32 hooks on future
predictions/target features ensure that BF16 compute retains FP32 MSE/cosine
reductions. Baseline output conversions are identity operations.

`report.json` contains completed cases, errors, per-experiment median speedups,
individual repeat speeds and numerical differences. A case error is retained and
does not become a claimed speedup. Finite short-run losses and gradients do not
establish policy quality or justify changing production. `result.json` and the
first fusion gradient are also saved separately per case. No rollout is run.

The job requests oneH200/32CPU/384GiB, gpu,gpux at their queried common maximum
walltime, requeue eligibility, per-run locking and nice10000 so formal jobs keep
priority. Completed cases survive continuation/requeue. PyTorch/Inductor/Triton
caches live under `/public/home/wangzihao/.cache/s1p/`. A separate inert tensor
reserves16%of GPU memory for the requested15% occupancy threshold; its allocation
is not model memory and does not appear in checkpoint or optimizer state.
