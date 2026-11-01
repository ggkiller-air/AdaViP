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

## Submitted five jobs

- Synced implementation and audit to GitHub branch
  `experiment/single_task_teacher_batch8_20261030`; frozen source commit
  `852e624`. All697copied files (including pinned Diffusion Policy) were
  SHA256-verified and made read-only. Source inventory/revisions and both
  pytest logs are in the run root. Submodule revisions are unchanged.
- Submitted B412402, C412403, D412404, E412405, dropout412407. Independent
  scheduler audit confirms each requests one NVIDIAH200,32CPUs/384GiB,
  gpu,gpux, queried common48h and Requeue=1. An allocated job's Partition
  resolves to its chosen partition; original dual-partition requests remain
  recorded in each queue JSON. `submissions.json` and `scheduler_audit.json`
  preserve IDs, resources, commands and logs. All five launch configs were
  rechecked against the saved original L1 config after submission; resume=null.
- At19:15CST B is RUNNING onserver13, loading the full RAM cache for its
  independent batch64 preflight; stderr is empty. C/D/E/dropout are PENDING /
  AssocGrpGRES. No formal optimizer update or passed GPU preflight claimed yet.
  Before GPU work, both allocation-local and independent SSH nvidia-smi reads
  show NVIDIAH200143771MiB,0MiBused/0%utilization and no compute process on
  the allocated device. Existing CPU interactive404454 remains unchanged.

## User requested direct training without preflight

- Explicit user instruction supersedes automatic preflight for these five runs.
  Removed automatic Future preflight from new batch scripts and dropout's
  automatic preflight call; the shared step wrapper skips IPC tests in Future
  and dropout training contexts. Already queued Future batch scripts retain
  their submitted shell body, so the sanity entry point returns immediately
  when S1F_PHASE=train. Standalone diagnostic helpers remain manually callable.
- Preserved all five job IDs and queue positions. At19:24CST B412402 already
  completed its earlier preflight and is in formal epoch2/global_step1120;
  leave its trainer and optimizer state running. C/D/E/dropout remain queued
  and will enter formal training directly. No training config changed.
- Preserved the full previous697-file source snapshot and inventory as
  `source_before_direct_training/` and `source_before_direct_training_sha256.json`.
  Atomically replaced only four orchestration files in the runtime source,
  restored read-only permissions, and updated source_sha256.json.
  `direct_training_update.json` records exact before/after hashes. Model,
  dataset and Future continuation fingerprints are unchanged; dropout has
  not started and will use the updated runner fingerprint from initialization.
- Python AST parsing, shell syntax and whitespace checks pass. No new GPU
  preflight or full test suite was run for this orchestration-only change.
