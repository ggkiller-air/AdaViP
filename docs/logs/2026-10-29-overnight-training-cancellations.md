# Overnight L1/L2/S1 cancellation audit

- Read-only audit at approximately 11:22 on 2026-10-29. `sacct -S
  2026-10-28 -j 397055,397057,397371` records all three as `CANCELLED by 0`:
  L1/397055/server51 and L2/397057/server52 at 00:56:07; S1/397371/server13
  at 05:41:06. Runtime was 3:24:43/3:24:43/6:04:57 against 100h/100h/48h
  limits. Stderr contains scheduler cancellation, no Python traceback or CUDA
  OOM. UID0 identifies the cancellation authority; accounting alone cannot
  distinguish an administrator from a privileged automation or its motivation.
- Original sibling queue cancellations are by UID2116 at allocation startup;
  these are expected duplicate prevention and distinct from the overnight stops.
  No replacement L1/L2/S1 allocation is currently queued. S1's `--requeue`
  eligibility does not itself resubmit a cancelled job.
- Existing latest checkpoints are readable. L1/L2 both retain global_step1,
  epoch0/batch_in_epoch1; neither reached steady-state training before the
  cancellation, with stdout still inside decoded RAM-cache loading. Metadata
  was also checked directly from the checkpoint pickle records.
- S1 latest preserves epoch36/batch_cursor504/global_step71904, including model,
  EMA, optimizer, scheduler and RNG. This matches its final logged training
  interval; fixed checkpoints E1/6/10/20 remain present.
- S2/397373, S3/397375 and S4/397376 remain RUNNING, with metrics advancing
  around epochs66/62/65. Separate existing Transformer continuation398812 is
  also RUNNING. No job was cancelled, changed or restarted during this audit.
- Artifact roots: language conditioning
  `/data/wangzihao/outputs/manifeel/language_conditioning_t2_batch64_ep500_seed42_20261028_204459`,
  encoder ablations
  `/data/wangzihao/outputs/manifeel/stage2_encoder_ablation_t2_batch32_ep600_seed42_retry_20261028_233606`.
  Scheduler evidence remains in `/data/wangzihao/outputs/manifeel/slurm/`.

## Cache loading follow-up

- L1 cache loading started at 21:32:12 and remained incomplete for 3:23:55
  before cancellation; 45/63 arrays totaling193.05GiB emitted successful load
  events. L2 started at21:32:45 and remained incomplete for3:23:22; only the
  tiny task0 state/action arrays emitted completion events. These counts measure
  fully loaded and hashed arrays, not partial reads of in-flight arrays.
- The same-night S1/S2/S3/S4 full299,494,742,868-byte cache loads finished in
  1258.79/1259.62/685.24/627.76seconds (20.98/20.99/11.42/10.46minutes).
  Thus L1/L2 startup time is anomalous, not normal cache preparation duration.
- Loader implementations differ: L1/L2 use `np.load(..., mmap_mode=None)`;
  Stage2 uses read-only mmap followed by `np.array(..., copy=True)`. Both hash
  complete arrays with eight threads. This is a candidate performance difference,
  not a verified cause; historical node I/O, RAM pressure and CPU scheduling
  telemetry is unavailable in these artifacts. GPU monitoring for L1/L2 sampled
  0% utilization throughout the long cache interval. No performance experiment
  or restart was launched during this follow-up.

## QoS verification

- Accounting reports `Account=yangyc,QOS=user_wangzihao` for L1/397055,
  L2/397057 and S1/397371, also for the surviving S2--S4. None used `overload`.
- Current QoS definitions: `user_wangzihao` may preempt `overload`;
  `overload` has `PreemptMode=requeue`, `GraceTime=00:03:00`, `NoReserve`.
  These match the user's description of opportunistic capacity, but do not
  identify these particular cancellations as overload preemption.
- Cluster configuration has `JobRequeue=0`. S1 explicitly requests `--requeue`;
  the frozen L1/L2 batch script does not. This is a recovery limitation to fix
  before any future use of opportunistic QoS, not evidence for this incident's
  cancellation cause. No submission/configuration changes made during audit.

## Authorized mmap transport repair and resubmission

- Changed the shared RAM loader to `np.load(..., mmap_mode="r")` followed by
  `np.array(mapped, copy=True)`, matching Stage2 transport. Full SHA256, shape,
  dtype and source-contract validation remain intact; RAM arrays are owned and
  read-only before fork. Added owned-RAM versus mmap assertions to existing
  all-window/normalizer parity coverage. S2--S4 remain on their frozen sources.
- L1/L2 batch script now requests `--requeue` and creates its IPC TMPDIR inside
  the srun step, addressing separate batch/step shared-memory mounts.
- Full CPU suite: `pytest -q --capture=sys`,259passed/1skipped/5existing warnings,
  261.67seconds. Shell syntax and `git diff --check` pass. Evidence:
  `/public/home/wangzihao/.cache/lc/requeue_20261029/`.
- Created isolated read-only source at the original L1/L2 root's
  `resume_mmap_20261029/source`. Verified all594original source-file hashes;
  only decoded_sensor_cache.py and train_language_conditioning.sbatch differ.
  The tested repository's runtime source fingerprint equals this frozen source.
- Migrated only checkpoint `data_provenance.source` from `c05b045f...512f2`
  to `5a49934b...658a2`. Compared every serialized model/EMA/optimizer tensor,
  all pickled training-state bytes and config after reload; unchanged except
  for this explicit provenance migration. Both cursors remain epoch0/step1/
  batch1. Original checkpoint inodes retained under
  `resume_mmap_20261029/original_checkpoints/`; exact migration script, manifest
  and per-variant verification reports are preserved beside the new source.
  Runtime resume validation remains strict. Scientific settings are unchanged.
- Requeued S1 from epoch36/batch504/step71904 using its original repaired source:
  gpu399040/gpux399041. Requeued L1 gpu399053/gpux399054 and
  L2 gpu399055/gpux399056 from migrated latest checkpoints. All use normal
  `user_wangzihao`,32CPU/384GiB/oneH200, requeue eligibility and per-run locks
  with sibling-only cancellation. Submission helpers queried current maximum
  walltimes: gpu100hours, gpux48hours. Separate submission records remain under
  each run's queue or the L1/L2 restart directory.
- At11:35 all six replacement requests were PENDING/Priority; no new cache-load
  duration or GPU training result is claimed. Existing S2/S3/S4 and Transformer
  jobs remain running, untouched.

## S2--S4 throughput and CPU affinity diagnosis

- At11:42, S2/S3/S4 were at epochs68/64/67 with138320/130160/136200steps.
  Recent completed-epoch training times were612--622/628--636/611--618seconds;
  corresponding data-wait fractions23.6--24.8%/6.1--7.3%/6.3--8.0%.
  These timers exclude checkpoint serialization after epoch completion.
  Last40GPU monitor samples (approximately20minutes) averaged40.8/44.5/44.8%
  utilization. No training exception; counters continue advancing.
- Confirmed live CPU affinity via /proc on the allocated nodes: S2 main2694457
  is confined to26,122; S3 main2696789 to64,160; S4 main2089872 to5,101.
  All eight workers per job alternate between these same two logical CPUs.
  These are sibling threads of a single physical core. Batch/srun parents have
  the full32logicalCPUs. S2/S3 step cpusets also permit their full32CPUallocation,
  so the restriction is process affinity, not a shortage of allocated cores.
- The frozen srun commands specify --cpu-bind=cores without an explicit
  --cpus-per-task argument. All three stderr files warned that only2CPUs were
  visible before creating8workers. The observed affinity is a confirmed
  transport/CPU scheduling limitation; its speed impact requires a before/after
  measurement rather than inferring that it explains all GPU idle time.
- Diagnosis was read-only; no affinity, cgroup, checkpoint or active job was
  modified during this investigation.

## L1 IPC exit and isolated retry

- L1/399053 ran12:04:42--12:23:58, then FAILED(1:0). Its cache now loads in
 707.02seconds, but worker transport cannot create
  `/dev/shm/lc_399053/pymp-*`; DataLoader subsequently times out at180seconds.
  The cause of that directory disappearing is not established. L2/399055 loaded
  in1334.14seconds and continued training at approximately99samples/s.
- Replaced only failed L1's future batch wrapper: short persistent
  `/public/home/wangzihao/.cache/lc/i/JOBID`, created/tested inside the step with
  an actual two-worker tensor-transfer probe before constructing the trainer.
  Existing L2/S1/S2/S3/S4 and Transformer processes remain untouched.
- New immutable LC source under the original run root's
  `resume_ipc_20261029_124551/source`. Runtime fingerprint still exactly
  `5a49934b...658a2`; model/data/protocol source is unchanged. Original latest
  step2checkpoint hard-linked into the retry directory. No checkpoint mutation
  or source-fingerprint migration is needed for this shell/IPC wrapper change.
- L1 resubmitted as single combined-partition job399171 with48hours and requeue
  eligibility, from the same output/latest. At12:46 it is PENDING. Independent
  wrapper regression and actual persistent-directory worker IPC passed.
