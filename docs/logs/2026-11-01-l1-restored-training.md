# Restore original L1 settings for five fresh ablations

- User authorized five new runs (Future B/C/D/E and L1 input-dropout10) with
  original L1 shared hyperparameters. All start from seed42 initialization;
  no batch32/E100 checkpoint is continued. Existing artifacts remain intact.
- Resolved all five launch configs against the original run's actual
  `language_conditioning_t2_batch64_ep500_seed42_20261028_204459/l1/config.yaml`.
  Shared data/model/AdamW/EMA/loader/training/diagnostic settings match:
  batch64/E500, LR8e-5, betas(.95,.999), WD1e-6, eps1e-8, warmup500updates,
  cosine over557000updates, clip1, full400demos/71265anchors, task-balanced
  replacement sampling, no validation,8workers, fixed E50/100/200/300/400/500,
  rolling latest every1000updates and each epoch. Entrypoints reject drift.
- Match original L1's final FP32 `stage0` runtime: cuDNN TF32 on,
  deterministic off, matmul TF32/autotune off;32allocated CPUs explicitly
  passed to srun. New Future allocations run separate batch64 affinity,
  gradient/target-EMA/update/checkpoint preflight processes before fresh formal
  initialization; dropout's no-update preflight also uses real batch64.
- Keep Future branch settings and fixed weights from prior launch configs:
  B=.25332773311046375, C=.019863649098324596,
  D=.17650865668093083, E=.03004826433718963. Dropout stays90%full,
  10%uniform among nine removals. These are the intended experiment differences.
- New root: `/data/wangzihao/outputs/manifeel/l1_restored_batch64_ep500_seed42_20261101/`.
  `protocol_audit.json`, `original_l1_config.yaml` and five `launch_config.yaml`
  files record the comparison. Idempotent submitter prepares/audits all five
  before queueing any; per-run locks and checkpoint continuation are retained.
- Initial scheduler: only user CPU interactive404454 is active; server30 has
  no CUDA device. Query partition maxima at each submission (observed gpu100h,
  gpux48h); request exactly one gpu,gpux/common48h/requeue H200 job per group,
  32CPUs/384GiB. GPU telemetry and compute-process inspection precede GPU work
  in each allocation. No pre-existing job changed.
- Focused regression:35passed/1CUDA-skipped in102.32s. New rejection coverage
  includes batch, LR, epoch budget, warmup, snapshot schedule, EMA, dataset seed
  and inference-step drift. Full suite:330passed/2skipped/5existing Transformer
  warnings in201.95s. Shell syntax and diff whitespace checks pass. No GPU
  preflight result is claimed at this preparation entry. Logs:
  `/public/home/wangzihao/.cache/l1r/`.
