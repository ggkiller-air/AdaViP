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
