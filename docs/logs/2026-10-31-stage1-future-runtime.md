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
