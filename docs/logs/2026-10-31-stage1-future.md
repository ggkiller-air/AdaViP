# Stage 1 L1 future-supervision experiments

- User requested B--E formal training and A--E sanity checks, all from the same
  random L1 initialization: t+4 single-frame T or V+T prediction, raw64 MSE or
  512D EMA-feature cosine, batch32/E100, snapshots20/40/60/80/100. No Stage2 or
  rollout. Work stays in AdaViP-main; the temporary independent worktree was
  removed and the pre-existing log modification was preserved.
- Audited L1: pre-language z_t256 from five trainable ResNet18/GroupNorm streams
  and proprioception; language512 concatenated through768->256->256. AdamW8e-5
  covers all online ResNet parameters. Original objective is Action Loss only.
- Independent future modules keep inference policy/EMA unchanged. Each sensor
  has a separate head. Latent target EMA starts from online initialization,
  uses fixed0.996 and handles parameters plus floating/integer buffers. Future
  frames are separately indexed and invalid episode tails are masked.
- Focused regression:19passed in50.95s, including exact action parity, common
  initialization, episode alignment, gradients through all five online image
  encoders, no language bypass, target no-grad/buffers, and exact A--E
  optimizer/policy/target-EMA continuation. Full-repository testing in progress.
- CPU host server30 has no GPU. Only user job404454 was present at inspection;
  no existing job was changed. Current partition maxima gpu100h/gpux48h;
  submissions must use gpu,gpux and common48h. GPU sanity and formal training
  have not yet been submitted at this log entry.
- Environment: manifeel Python3.8/PyTorch2.4.1+cu121. Pytest FD capture fails
  when truncating anonymous temporary files on the persistent filesystem;
  `--capture=sys` works. Test TMPDIR is `/public/home/wangzihao/.cache/s1f/t`.

## B--E-only clarification and GPU checks

- User clarified that A already exists and needs neither GPU sanity nor formal
  training. Cancelled only our new sanity job407844 during cache loading,
  before any optimizer update; no existing experiment was changed. Updated
  the driver to B--E only and submitted407848 on gpu,gpux/common48h.
-407848 started on server13 with one NVIDIA H200143771MiB, initially0MiB/0%
  and no compute process;32allocated CPUs. Persistent-directory DataLoader IPC
  passed. Cache loading precedes four-group GPU checks and calibration.
- Initial full-policy SHA256 is identical across groups:
  `21b442e6436d5e0ed7d900c8f5838a8c5cb4519576e4ca4eafe9f74095125228`.
  Policy313011655parameters; B/C/D/E heads1252454/788480/3131135/1971200;
  C/E frozen target encoders22353024/55882560. Other groups have no target EMA.
- Root: `/data/wangzihao/outputs/manifeel/stage1_future_k4_batch32_ep100_seed42_20261031_133616/`.
  Frozen `source_bcde/` includes pinned Diffusion Policy and SHA256 inventory.
  Formal jobs are not yet submitted; report successful preflight and weights
  before starting them.
- Full regression with `OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1`
  and `pytest -q --capture=sys`:272passed/1skipped before interruption at an
  existing FileLock wait in the Stage2 cache test. That test independently
  passed; the remaining27tests then passed in24.62s. Thus all299collected
  tests passed across the two runs, plus one collection skip. Evidence:
  `/public/home/wangzihao/.cache/s1f/{full_pytest,remaining_pytest,isolated_stage2_cache}.log`.

## GPU pass and formal submissions

-407848 completed all four B--E checks, ExitCode0:0,9m29s. Full279GiB cache
  loaded/hashed in387.16s. All eight workers inherited32CPUs and were separately
  bound to48--55. All four batch32 policies had identical initialization;
  exact original Action Loss parity, nonzero Future gradients to all five
  ResNets, independent heads, finite losses/gradients, actual optimizer updates,
  correct target EMA/no-grad and multi-GB checkpoint writes passed.
- Calibrated over8batches without optimizer/EMA updates. Fixed weights:
  B0.25332773311046375, C0.019863649098324596,
  D0.17650865668093083, E0.03004826433718963. Weighted fusion-gradient ratios
  were approximately0.091--0.118. Peak GPU allocation B/C/D/E:
  14.122/14.245/14.165/14.462GiB. Preflight evidence is `sanity/report.json`.
- Reported the successful checks and final weights to the user before formal
  submission. Source implementation pushed to GitHub main atfb0142d.
  Submitted exactly four combined-partition/common48h/requeue jobs:
  B407886, C407887, D407888, E407889. No A job or Stage2/rollout was submitted.
- At13:49CST B407886 was RUNNING onserver13/H200 with32CPU/384GiB; C--E
  PENDING/AssocGrpGRES. B completed formal step1 with Action1.1925285,
  Future0.0168204, Total1.1967895, weighted fusion-gradient ratio0.08935;
  latest.ckpt4.7GiB exists. Shared initialization SHA256 matches the old L1
  common hash5481f6ca...18214. Startup's cold raw-data step is saved before
  full RAM loading; its3.6samples/s is not steady-state throughput.
- Confirmed all launch configs: RAM decoded cache, pin_memory=true,
  8persistent workers/prefetch1, nonblocking H2D. Runtime is the previously
  validated tf32_deterministic profile: cuDNN TF32=true, benchmark=false,
  matmulTF32=false, FP32. Actual32-CPU trainer affinity is recorded in
  `B/runtime_407886.json`; production worker affinities are logged separately.
- Per-group paths under the run root: `B/`--`E/launch_config.yaml`,
  `config.yaml`, `progress.jsonl`, `metrics.jsonl`, `diagnostics.jsonl`,
  `preflight_calibration.json`, `checkpoints/`. `formal_submissions.json`
  records IDs/configs/weights. Slurm stdout/stderr are in the common
  `/data/wangzihao/outputs/manifeel/slurm/s1f_[B-E]_k4_batch32_ep100_JOBID.*`.
