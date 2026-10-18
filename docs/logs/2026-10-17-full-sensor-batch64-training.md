# Full-sensor batch64 first training (2026-10-17)

- User requested an actual training submission followed by evidence-based
  batch/resource adjustment. Kept old experiments and `third_party/` unchanged.
- Added20-step progress metrics, window throughput/data wait/gradient/condition
  counts, first-step100 and periodic1000-step checkpoints, signal-safe stop,
  sampler cursor and partial-epoch statistics. CPU regression35 passed34.90s,
  including bitwise resumes aftersteps1/2/3 and identical offset sampler order
  for0/2 workers. Command: project ManiFeel Python, OMP/MKL threads1,
  `PYTHONPATH=$PWD:$PWD/third_party/diffusion_policy`, `-m pytest -q
  tests/test_full_sensor_stage0.py
  --basetemp=/data/wangzihao/pytest-tmp/full_sensor_longrun_20261017`.
  Output `/data/wangzihao/outputs/manifeel/slurm/full_sensor_longrun_tests_20261017.log`.
- Login allocation server37 has no visibleGPU. Left interactivejob366142
  untouched. Initial engram-partition submission rejected with invalid
  account/account-partition; current associations yangyc/overload, engram
  permits neither. Used gpu partition as existing H200 training launchers,
  under user's explicit sbatch training authorization; no interactive training.
- Submittedjob367952, oneH200,16CPU,64GiB,6h limit, USR1 warning300s.
  Target300 epochs, train/val batch64,69,205 train samples/epoch,
  1,082 steps/epoch,324,600 total updates. Full nine-task validation each epoch;
  no max-step truncation. LR8e-5,warmup500,80%full/20%nine-removal,EMA,seed42.
  No claim that300epochs fits this first6h allocation; exit saves cursor/RNG.
- Source frozen (with SHA256/Git metadata) before submission under
  `/data/wangzihao/outputs/manifeel/full_sensor_reference_batch64_ep300_seed42_20261017_130045/source`.
  Third-party code referenced read-only; no Git commit/push. Run output uses
  that directory's parent. Resume by resubmitting same launcher/source/run
  with `ADAVIP_RESUME=.../checkpoints/latest.ckpt`; protocol changes rejected.
  Initial state PENDING. Logs:
  `/data/wangzihao/outputs/manifeel/slurm/full_sensor_dp_batch64_ep300_367952.{out,err}`.
- At13:04:28+08:00 job remainedPENDING/Priority, not yet running; no batch64
  throughput, GPU memory or validation claims made. Source SHA256 verification
  and shell/whitespace checks passed. Full suite169 passed,1 skipped, same old
  FiLM/CLIP source-asset test failed (51.13s). Output:
  `/data/wangzihao/outputs/manifeel/slurm/full_sensor_training_all_tests_20261017.log`.
  Queue status: `squeue -j 367952`; after allocation inspect progress.jsonl,
  gpu_monitor.log and all-nine-task metrics.jsonl before choosing a new batch.

## IPC startup failure and replacement (13:30–13:35+08:00)

- On user status check, job367952 was RUNNING onserver44 since13:12:14 but
  stalled before any confirmed optimizer progress: no progress.jsonl or ckpt,
  repeated `OSError: AF_UNIX path too long` in worker tensor FD transport.
  H200143771MiB,2951MiB occupied,0% utilization for successive monitor samples.
  Cancelled onlyjob367952 (CANCELLED,18min49s); interactive366142 untouched.
- Bounded overlap diagnostic on its allocation showed `TMPDIR=/tmp` but
  `tempfile.gettempdir()` falling back to cwd. The frozen source cwd is long
  enough to overflow Unix-socket paths. No CUDA OOM/batch64 result established.
- Launcher now exports short writable `/data/wangzihao/t/mf_JOBID`, verifies
  actual tempfile selection/socket creation/two-worker tensor transport before
  model startup (90s outer bound). DataLoader timeout180s prevents unlimited
  waits; zero-worker loaders resettimeout0. Home filesystem anonymous-tempfile
  truncate failed in a local probe, so selected the verified data-root tempdir.
- IPC probe succeeded (`socket_path_bytes=49`,two workers). Focused regression
  including runtime checks and previous Stage0 tests:38 passed43.00s with
  `TMPDIR=/data/wangzihao/t`, projectPython, OMP/MKL1, samePYTHONPATH;
  `-m pytest -q tests/test_full_sensor_runtime.py tests/test_full_sensor_stage0.py
  --basetemp=/data/wangzihao/pytest-tmp/full_sensor_ipc_fix_20261017`.
  Output `/data/wangzihao/outputs/manifeel/slurm/full_sensor_ipc_regression_fixed_20261017.log`.
- Fresh frozen source/output, preserving failed run unchanged:
  `/data/wangzihao/outputs/manifeel/full_sensor_reference_batch64_ep300_seed42_ipcfix_20261017_133441`.
  Submitted replacementjob368013 at13:34:42, sameH200/batch64/epochs300/6h/
  LR/EMA/dropout/data split; no checkpoint existed to resume. At13:35:03 still
  PENDING/Priority. New Slurm logs use suffix368013; real GPU training remains
  unverified until allocation preflight and optimizer progress appear.
- At13:40:24 replacementjob368013 was RUNNING onserver09 (elapsed5min03s).
  Allocation IPC preflight passed with59-byte socket path and2-worker tensor
  transport; stderr empty. Actual optimizer progress reachedstep60/1082 of
  epoch1 (3,840 samples); all nine removal conditions appeared, full3045/3840.
  Latest20-step loss0.76241, finite gradient norm4.50779, throughput21.15
  samples/s, data-wait fraction92.46%. Torch peak allocated24.78GiB;
  nvidia-smi monitor reportedH200143771MiB total/28109MiB used, sampled0%
  utilization. This indicates a data-feeding bottleneck, not batch64 OOM;
  retainbatch64 rather than blindly increasingbatch128. No checkpoint yet
  (first scheduledstep100), no completed epoch/validation or quality claim.
