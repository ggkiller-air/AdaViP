# Move all current training to normal group quota

- User requested long-running jobs use normal group quota instead of borrowed
  overload GPUs, stop unnecessary CPU checking jobs, and maintain more than
  15%VRAM on every allocated GPU. At migration all nine GPU jobs were pending;
  no active optimizer update or label scoring was interrupted.
- All nine now use Account=yangyc/QOS=user_wangzihao, one H200 each,
  gpu,gpux common48hour maximum, Requeue=1. Explicit future batch defaults
  use the same normal account/QOS. No overload GPU job remains in the queue.
- Preserved DP job IDs423348(ManiFeel) and425476(Tabero E150) by changing
  pending account/QOS in place. Their existing allocation-local16%memory
  holders remain active parts of their launchers, including on requeue.
- Replaced seven held native jobs with normal-account launchers while
  preserving original source/config/output/Teacher/labels/checkpoints:
  insert_hole425381→426113; insert_tube425370→426114;
  lift_bottle425374→426115; lift_can426088→426116;
  pull_out_key426071→426117; put_bottle426089→426118;
  Tabero-VTLA426072→426119. New jobs were held until registry/account/QOS
  verification completed, then released together. No duplicate run remains.
- New native shell starts its own independent zero-filled16%VRAM holder,
  verifies reservation identity/fraction before loading the model, monitors
  the holder from inside the GPU allocation, and requests checkpoint/requeue
  if it exits. Memory is outside model, optimizer, RNG and saved weights.
  Soft stop/walltime requeues the same job, preserving the normal account and
  memory launcher without a separate recovery supervisor. Actual GPU model,
  UUID/memory/utilization/process preflight remains inside the allocation.
- Cancelled CPU recovery423864 and VRAM supervisor423254; sacct confirms
  both CANCELLED by2116. Retained lightweight five-minute table refresher
  423354 and left the user's interactive CPU allocation425310 untouched.
  Registry records allow_overload=false and recovery_job_id=null.
- Validation: every queued job reports yangyc/user_wangzihao,48h,Requeue=1;
  saved checkpoint/completion-marker byte sizes, mtimes and inodes match
  the pre-migration snapshot. Both original native source inventories
  remain unchanged(FTP251files; Tabero252files). All three launchers pass bash syntax.
  Training/model code and data were not changed. All nine remain pending;
  live GPU VRAM measurements await allocation and are not claimed here.
- Migration receipt, original scheduler metadata, source launcher and holder:
  `/data/wangzihao/experiments/advanced_uvip/20261103_1707/normal_queue_migration_20261104_124239/`.
  Canonical root progress Markdown remains a table and update time only.
