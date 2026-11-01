# Separate runtime benchmark and VRAM occupancy

- User requested >=15% VRAM occupancy for current runs and an independent
  runtime queue for experience; this round's formal training settings and
  calibrated Future weights must stay fixed.
- Submitted CPU supervisor407906 (cpu/1CPU/1GiB/5days), reusing the existing
  `reserve_teacher_gpu_memory.py` with a four-run manifest adapter under the
  formal root's `memory_supervisor/`. It follows only B--E and their continuation
  queue files, attaching a separate zero-filled16% CUDA tensor to running jobs.
- Verified B407886 holder336756 on server13/H200, reserved24017601822bytes;
  independent nvidia-smi read40971/143771MiB (28.50%) after reservation. No
  model/optimizer/checkpoint/RNG state was changed. C--E awaitGPU quota and will
  receive holders when started. Other jobs were untouched.
- B entered full-RAM steady training, step1960/epoch1 at14:00CST, approximately
  157samples/s and8.6% epoch-aggregate loader wait (includes cold startup).
  Action.0371015/Future.000682832/Total.0372745, finite; worker/pin/TF32 settings
  remain the original approved production profile.
- Added independent benchmark script, Slurm wrapper, plan and4helper tests.
  B/E workloads,9profiles,2randomized repeats, batch32,24warmup+120timed updates.
  Compare BF16, channels_last, cuDNN autotune, matmulTF32, fusedAdamW, U-Net
  compile and combinations. Record throughput/compile cost/memory/loader wait,
  first losses and fusion/ResNet gradients versus baseline. No production
  checkpoint is read; no production directory is writable by the benchmark
  interface. Full cache/RNG/init/EMA steps remain controlled across profiles.
- Four helper tests passed in7.91s; Python compilation and shell syntax passed.
  GPU execution must wait for its own allocation; no performance gain claimed.

## Independent benchmark submitted

- Final helper recheck4passed in6.04s. Committed/pushed implementation to main
  at5ff79ee. Frozen source and SHA256 inventory are stored at
  `/data/wangzihao/outputs/manifeel/stage1_future_runtime_benchmark_20261031_140201/`.
- Submitted407932, nice10000, oneH200/32CPU/384GiB, gpu,gpux/common48h with
  requeue and per-output lock. Currently PENDING. Benchmarks read only the
  production launch configs and sanity initialization metadata, create fresh
  models, and write exclusively to the separate benchmark root. Formal B--E
  code/config/checkpoints/loss weights were not changed.
- Main future weights remain B.25332773311046375, C.019863649098324596,
  D.17650865668093083, E.03004826433718963. The shared-gradient contribution
  target is approximately10% from the previous eight-batch calibration.
- Results will be `report.json`, with per-case `result.json` and reference
  gradients. CPU supervisor407906 continues to guard formal B--E memory;
  benchmark407932 starts its own independent16% holder when allocated.

## Completed runtime results

-407932 COMPLETED,57m56s,ExitCode0:0. All36cases passed,0errors; both workloads
  shared one initialization hash and every case had8distinct worker CPU masks.
  Full cache load/hash76.03s. Production settings were not changed.
- Median samples/s (B/E) and change relative to each workload's baseline:
  baseline180.48/149.47; fusedAdamW190.50/159.39(+5.55%/+6.64%);
  compileU-Net182.41/164.59(+1.07%/+10.12%);
  autotune182.80/156.14(+1.28%/+4.46%);
  matmulTF32179.55/156.32(-0.52%/+4.59%);
  BF16165.56/144.95(-8.27%/-3.03%);
  channels_last167.70/147.19(-7.09%/-1.52%);
  BF16+layout155.61/137.40(-13.78%/-8.07%);
  BF16+layout+fused+compile172.64/144.00(-4.35%/-3.66%).
- FusedAdamW improved both paired repeats: B+5.63/+5.47%, E+10.30/+3.17%.
  Its initial loss matched baseline; initial fusion gradient relative difference
  was zero or <=3.5e-7. This is the most consistent candidate measured here.
- Compile was variable: B-2.51/+4.69%, E+18.36/+2.31%. First B compilation
  warmup319.80s versus baseline35.47s; cached compile warmups62--65s. No
  blanket10% compile gain is established from two repeats.
- BF16 first-loss relative difference <=0.0356%, initial fusion gradient cosine
  approximately0.99994--0.99995, relative gradient error1.0--1.1%; all updates
  finite, but no throughput benefit in this stack/batch32. Most loader waits
  were about0.1%; no operator profile was taken to attribute the slowdown.
- Recomputed initial-gradient cosine in float64: raw float32 cosine reports
  slightly exceeded1 even for self-comparison. Preserved raw reports; derived
  numerical checks in `analysis_summary.json` and readable table in `results.md`
  under the benchmark root. These are update timings excluding checkpointing/
  epoch diagnostics, with no rollout or policy-quality claim.
