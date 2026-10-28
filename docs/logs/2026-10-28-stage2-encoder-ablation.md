# Stage2 Encoder Training Strategy Ablation

- Implemented approved S1--S4 on the existing Stage2 policy/loss/optimizer/data
  helpers, with one ablation trainer. S1 freezes only five ResNets; S2 trains
  them from C at1e-5. Their other tokenizers retain C/8e-5. S3/S4 freeze the
  complete E500 context path, and train independent Action ResNets, temporal
  projections, P/L and Sensor Identity from B at8e-5, with the original500-update
  encoder ramp shifted to B. S4 randomizes only ResNets/temporal projections
  while preserving the seed42 common RNG stream. S0 defaults/results and L1/L2
  source/jobs are unchanged.
- Protocol: batch32/600epochs/seed42,65,265 complete K64 windows, original
  task-balanced replacement sampling, losses, AdamW, global clipping and EMA.
  Fixed checkpoints E1/6/10/20/50/100/200/300/400/500; rolling latest reaches E600.
- First raw batch checks B/C gradient routes without optimizer updates and
  restores RNG, then performs the next formal update and saves latest before
  loading/hash-checking all299,494,742,868 decoded bytes with8threads. Afterwards
  use8persistent fork workers, read-only shared RAM, separate core affinity,
  pinned batches and nonblocking H2D. Count-weighted Overall/Per-task Action and
  Utility losses, throughput, loader wait, CUDA allocated/reserved peaks,
  parameter counts, encoder gradients and GPU telemetry are persisted separately.
  Frozen raw/EMA weights are checked before every checkpoint.
- CPU:12new ablation tests pass, including all four exact model/EMA/Adam/RNG/
  sampler resumes, numerical inference restoration, fixed Teacher z_t, frozen
  weight tamper detection, raw/RAM parity and persistent-worker equivalence.
  Full suite:257passed/1skipped, five existing Transformer warnings. CLI help,
  shell syntax, diff checks and mock dual-partition submission also passed.
  Evidence: `/public/home/wangzihao/.cache/s2a/evidence/`.
- Actual E500/K64/cache metadata check passes. Four common initialization SHA256
  values match `f0741c79e40edf77e27152fde25673a9b1d5c9f21c09b8846886b5a3dfe83aae`.
  Total parameters including fitted Normalizer:314,041,736(S1/S2),371,905,224(S3/S4).
  S3/S4 B/C trainable parameters313,513,232, with no shared branch storage.
- Test filesystem pitfall: pytest FD capture hit an unlinked-temporary-file
  truncate error; `--capture=sys` avoids it. An initial suite was interrupted
  while waiting on FileLock; the complete verbose rerun passed without source
  changes to existing tests or experiments.
- Read-only GPU inventory before submission: job392551/server13 has H200,
  143771MiB total/38071MiB used/99% utilization, PID3166774. L1/L2 jobs397055/
  397057 run on server51/server52. No existing job was modified or canceled.

## Formal submissions

- Frozen code commit `3ada9b9`, independent branch
  `experiment/stage2_encoder_ablation_t2`; L1/L2 branch remains at `580766f`.
  Source and pinned Diffusion Policy package have601SHA256-inventoried files,
  made read-only under
  `/data/wangzihao/outputs/manifeel/stage2_encoder_ablation_t2_batch32_ep600_seed42_20261028_221516/source`.
  CPU evidence is copied into the sibling `cpu_evidence/` directory.
- S1 gpu397198/gpux397199; S2 gpu397200/gpux397201;
  S3 gpu397202/gpux397203; S4 gpu397204/gpux397205. Each requests one H200,
  32CPU and384GiB. Both current maximum walltimes were queried during each
  submission: gpu100hours, gpux48hours. Per-strategy lock and queue files
  restrict cancellation to that strategy's sibling allocation.
- Initial scheduler inspection: all eight PENDING/AssocGrpGRES. No formal
  update, new GPU measurement or training throughput result is claimed yet.
  Existing jobs397055/397057/392551 remain RUNNING and untouched.
- Each strategy's sibling `s1/`--`s4/` directory holds `submission.txt`, queues,
  then metrics/gradient/runtime logs and checkpoints. Scheduler logs will be
  `/data/wangzihao/outputs/manifeel/slurm/uvip_sN_t2_batch32_ep600_JOBID.out`
  and `.err`; the root `submissions.json` maps strategies to both job IDs.

## Startup failure repair

- The four actual allocations failed before any formal update:397198/397203/
  397205 raised `Stage2UtilityDataset.task_specs` AttributeError;397200 could not
  create its multiprocessing directory under the step's `/dev/shm`. All sibling
  cancellations were expected duplicate prevention. No latest checkpoint existed.
- Expose the source's task metadata on the real label wrapper. Create TMPDIR
  inside the srun step before starting Python. Added real-wrapper training/RAM/
  per-task logging regression and an actual missing-directory IPC subprocess
  regression, rather than relying only on ToyDataset. Added `--check-inputs` to
  run the real CLI's production dataset/IPC/first-batch validation on CPU.
- CPU checks:14ablation tests plus15existing policy/MLP/evaluation tests pass.
  The actual K64 CLI check passes:65265windows,9tasks, batch32, five T2 image
  streams, state[32,2,7], language[32,2,512], action[32,16,7], utility[32,9].
  Evidence: `startup_fix.log`, `startup_fix_existing.log`, `real_inputs_fix.log`
  under `/public/home/wangzihao/.cache/s2a/evidence/`.
- Enable Slurm requeue eligibility for scheduler-managed interruptions, retaining
  safe-stop continuation and stopping on application errors. Training losses,
  initialization, sampling, budget and checkpoint cadence are unchanged.
- GPU inventory before retry: existing392551/server13 remains H200143771MiB,
  38071MiB used/95% utilization, PID3166774; L1/L2 remain running and untouched.
