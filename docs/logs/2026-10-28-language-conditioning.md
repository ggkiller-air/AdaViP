# T=2 L1/L2 language conditioning

- User approved implementation and direct formal training after CPU checks,
  explicitly skipping GPU preflight and protecting Slurm job392551. Protocol:
  all400demonstrations/71,265anchors, no validation, task-balanced replacement
  sampling, batch64,500epochs, seed42, AdamW8e-5,500-update warmup/cosine,
  full inputs only. Fixed checkpoints50/100/200/300/400/500 plus rollinglatest.
- Added independent six-sensor ResNet18/GroupNorm and temporal projections,
  Concat+MLP1542→256→256, then L1MLP768→256→256 or L2MLP512→256→512.
  L2 final weight/bias are zero. Unchanged reference DP/U-Net receives256D;
  no Utility/Future/attention modules or language-sensitivity diagnostics.
- Both actual-size CPU constructions have identical common initialization SHA256
  `5481f6cac8c1a6cf624ce188e46c658f1b5067c446323ff74c1d740378c28214`.
  Total parameter counts313,011,655/313,011,911; heads262,656/262,912.
  Verified existing cache source contract without loading image arrays.
- First batch is a real formal optimizer update using raw data. Save atomic
  latest immediately, then8-thread load/hash the existing279GiB float32 cache.
  Resume training from the next draw using shared read-only arrays,8persistent
  workers, pinned host batches, nonblocking H2D, worker core affinity and Slurm
  core binding. Cache startup is timed separately from steady-state throughput.
- Raw fixed training loss probes32windows/task×4noise draws each epoch;
  fusion-MLP gradient probes8windows/task×4draws at epochs1/2 and every5.
  Persist ordered9×9cosines, zero-norm validity, gradient norms and negative
  unordered-pair fraction. Training loss/counts, cumulative counts, peak CUDA
  allocated/reserved memory, loader wait and GPU telemetry are separate logs.
- Eight focused CPU tests pass, including both variants' exact model/EMA/Adam,
  noise stream, sampler and count restoration; diagnostics leave RNG/gradients
  unchanged; first formal update precedes full cache loading. Full CPU run:
 244passed/1skipped/1runtime-environment failure caused by an overly long test
  TMPDIR. Recheck uses shorter persistent `/public/home/wangzihao/.cache/lc/tmp`;
  production IPC uses short `/dev/shm/lc_JOBID`. The three runtime tests then
  passed in12.15s; all245 distinct CPU tests passed across these runs,1skipped.
  No existing source was changed.
- Evidence: `/public/home/wangzihao/.cache/language_conditioning_20261028/`
  contains focused/full/runtime logs and actual_config.json. Submit helper reads
  current gpu/gpux maximum times, with per-run lock and sibling-only cancellation.

## Formal submissions

- Frozen training commit: `fac904e`; source plus pinned Diffusion Policy package
  copied under `/data/wangzihao/outputs/manifeel/language_conditioning_t2_batch64_ep500_seed42_20261028_204459/source`.
  All 594 source files have SHA256 inventory; source is read-only. Separate `l1/`
  and `l2/` siblings contain queues, logs and future checkpoints.
- Submitted L1: gpu `397055` (100 hours), gpux `397056` (48 hours).
  Submitted L2: gpu `397057` (100 hours), gpux `397058` (48 hours).
  Both maximum limits were queried at submission; no shortened time-min.
- Initial scheduler inspection: all four are PENDING (Priority/Resources).
  No formal optimizer update, GPU result or long-run throughput is claimed yet.
  Job `392551` remains RUNNING on server13; read-only sample before submission:
  NVIDIA H200, 143771 MiB total, 38071 MiB used, 93% utilization, PID3166774.
