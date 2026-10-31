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
