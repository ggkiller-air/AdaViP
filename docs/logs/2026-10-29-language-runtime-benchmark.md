# Authorized independent L1/L2 runtime investigation

- User explicitly authorized two independent training tests and, if improved,
  stopping only original L1/L2 and continuing from their checkpoints. S1--S4,
  original Transformer and other jobs are protected. Original LC jobs399298/
  399055 continue throughout measurement.
- Added `benchmark_language_runtime.py`, using the actual unchanged LC
  workspace update/EMA/optimizer/sampler/noise stream, full71265anchor dataset,
  verified299494742868-byte RAM cache and batch64. Each variant starts every
  case from its own hard-linked original checkpoint. Case order is randomized
  for two repeats; each has24warmup/160CUDA-synchronized timed updates.
- Five cases isolate observed two-thread strict baseline,32CPU strict,
 32CPU nondeterministic/TF32-off,32CPU deterministic/cuDNN-TF32-on, and
 32CPU original-Stage0-style cuDNN-TF32-on/nondeterministic. MatmulTF32,
  BF16 and autotuning remain off. Worker affinities and explicit flags are
  recorded. Scientific models/data/batches/loss/optimizer are unchanged.
- Candidate gate requires at least20% speedup over baseline in both repeats,
  mean loss no more than1.25x baseline and observed gradient norms no more
  than3x baseline. The CPU-only change is preferred if within5% of fastest.
  Selected candidate must additionally save an independent real checkpoint,
  restore through the strict provenance checks and advance another4updates.
  This short test is not evidence of identical long-run convergence.
- Added explicit runtime profile helper/CLI recording, strict by default.
  Future LC batch steps now explicitly request all allocatedCPUs. Runtime
  profile selection survives normal continuation via the exported environment.
  These entrypoint/shell changes leave the strict runtime source fingerprint
  `5a49934b...658a2` unchanged; no checkpoint mutation is needed.
- Frozen599file source and hard-linked snapshots at
  `/data/wangzihao/outputs/manifeel/language_runtime_benchmark_20261029_160233/`.
  L1 snapshot epoch9/step10026; L2 epoch17/step19000. Both provenance/source
  checks pass. Exact hashes, configs and submission commands are preserved.
- Queried gpu100h/gpux48h; submitted one gpu,gpux/48h/requeue job per variant:
  L1/399622(server26), L2/399623(server23), each oneH200/32CPU/384GiB,
  own lock/logs and verified persistent IPC. Initial GPUs were empty. Actual
  srun affinity exposes32logicalCPUs in both allocations. Cache loading is
  underway; no measured speedup or production restart is claimed yet.
- Validation: shell syntax, Python compilation and diff checks pass. Full
  CPU suite268passed/1skipped/2environment failures(282s): Gloo FileStore
  cannot lock the NAS test path; nested IPC temp path exceeds108bytes.
  Rechecking exactly those two tests with shared-memory rendezvous and short
  persistent TMPDIR passes both in82.10s. Thus all270distinct tests pass
  across these runs; five preexisting Transformer warnings remain. Focused
  runtime/submission checks10passed. Persistent logs:
  `/public/home/wangzihao/.cache/lc/runtime_tests_20261029/`.

## Initial GPU measurement

- L1 full-cache copy/hash completed in445.91seconds. First completed case,
  32CPU/cuDNN-TF32-on/deterministic, measured233.8samples/s over160updates
  after24warmup. This is a single observation; no production switch yet.
  The production progress field includes historical checkpoint timing state,
  so benchmark comparisons use the independent synchronized160-update timer.
  L2 full-cache loading remains in progress at16:14.

## Repeated L1 measurement and continuation gate

- L1/399622 completed its ten cases and real-checkpoint continuation gate at
  approximately16:36. Median samples/s: observed two-thread strict103.99;
  32CPU strict134.08; nondeterministic/TF32-off142.97;
  deterministic/cuDNN-TF32-on251.06; original-Stage0-style268.54.
  The fastest profile repeated268.59/268.49samples/s. CPU correction alone
  helps approximately29%, while compute flags account for the remaining gain.
- L2 full-cache load completed in584.93seconds. Its first fastest-profile
  measurements248.56/250.90samples/s also repeat. Its final baseline and
  checkpoint continuation check remain pending at16:36. Original jobs continue.
- Prepared an explicit per-variant handoff script at benchmark-root `rollover.py`.
  It refuses to run unless both reports pass speed/loss/gradient/resume gates
  and select the same profile. It pauses only the old batch shell to prevent
  old-profile automatic resubmission, requests the trainer's normal stop-file
  checkpoint, verifies final step/EMA/optimizer/RNG/provenance, then terminates
  old allocation cleanup and submits a dependent continuation with the new
  immutable entrypoint. All scientific state comes from the live production
  trainer's final checkpoint, never from test-updated model weights.

## Both gates passed and production handoff

- L2 ten-case medians: baseline100.87,32CPU strict130.18,
  nondeterministic/TF32-off140.31,deterministic/TF32-on238.29,
  Stage0-style249.76samples/s. Both variants select Stage0-style FP32
  parameters with cuDNN-TF32 on, deterministic off, matmul-TF32/autotuning off.
  Loss/gradient gates pass in both repeats. L1 real checkpoint10042resumes to
  10046; L2 checkpoint19016resumes to19020. Both independent jobs COMPLETED.
- Original L1 safely saved epoch13/batch189/global_step14671, preserving
  model/EMA/optimizer and Python/NumPy/Torch/CUDA/training-noise RNG. Source,
  normalizer and protocol provenance are unchanged; EMAstep equals14671.
  Original399298 was then explicitly cancelled by UID2116 at16:42:45.
  Replacement399733 started onserver30 at16:44:45, reports the selected flags
  and all32logicalCPUs, restores14671and saves formal update14672before RAM
  loading. No test weights or earlier checkpoint cursor were substituted.
- Original L2 safely saved epoch21/batch200/global_step23594 with the same
  state/provenance checks. Original399055 was explicitly cancelled at16:46:08.
  Replacement399747 started onserver23 at16:47:46, single gpu,gpux48h/requeue
  request with dependency and preserved per-run lock. Its startup is monitored.
- NAS negative lookup caching delayed visibility of L2's SAFE_STOP onserver21;
  direct writer-node inspection confirmed23594before the control-node helper
  independently saw and verified it. No checkpoint was rewritten. Stop-file
  requests are now also written on the trainer node to reduce this delay.
- During original L2 termination its immutable wrapper generated two legacy
  afterok continuations399746/399748. Both remained DependencyNeverSatisfied
  after399055was cancelled and never ran. Cancelled only these two new,
  obsolete children; recorded their source/dependency. Thus pausing the old
  batch shell did not completely prevent auto-submission during final signal
  cleanup. No pre-existing unrelated queue entry or running job was changed.
- Full handoff details and hard-linked before/final checkpoints are in
  benchmark-root `rollover_l1/` and `rollover_l2/`. Initial production startup
  has no stderr error. Full-cache reload and steady-state verification remain
  required before claiming achieved production throughput.

## First verified production performance

- L1/399733 restored14671and saved14672before reloading RAM in345.99s.
  At16:55:28 actual post-warmup production window15220--15380reports
  272.68pure-train samples/s and272.26wall samples/s over37.61seconds,
  with finite losses/gradient norms. This meets the old228samples/s reference
  in the actual continued trainer, not just the independent benchmark.
  Exact window samples and measurements are in `production_progress.json`.
- Pure-train window time is reconstructed by differencing cumulative epoch
  samples/reported throughput between two records in the same epoch; wall
  time uses record timestamps. This avoids mixing pre-handoff timing state
  preserved in the checkpoint and distinguishes checkpoint pauses. A previous
  window crossing step15000measured224.70wall samples/s because it includes
  the periodic full checkpoint write; subsequent compute window is faster.
- Independent allocated-GPU/proc inspection confirms H200,29877/143771MiB
  (20.78%), trainerPID1040006with32logicalCPU affinity and eight workers on
  distinct CPUs48--55. L2/399747 restored23594and saved23595before loading;
  its full cache and production speed are not yet verified at this entry.

## Verified production outcome, approximately17:00

- L1 continued normally through epoch14and into15. Latest independently read
  complete checkpoint at15596preserves model/EMA/optimizer/all RNG streams
  and unchanged data/normalizer/protocol/source provenance. Subsequent live
  window15840--16000measures272.73pure-train/272.31wall samples/s; next
  window16320--16480measures274.16/273.74samples/s. Losses/gradients finite.
- L2 full-cache reload completed in471.80seconds. Direct inspection inside
  its own allocation confirms32post-cache training records through24220,
  epoch22. Window24060--24220measures256.85pure-train/256.45wall samples/s
  over39.93seconds. Reader-node stdout visibility still lagged at four records,
  so writer/allocation observations are the primary evidence for this check.
- L2 H200 memory29879/143771MiB(20.78%); trainerPID1033384has32logicalCPUs,
  eight workers separately bound to4,19,20,21,25,26,28,29. L1 equivalent
  observations were recorded above. Both satisfy the reported15% occupancy
  floor without adding buffers. No recurring GPU filler compute was introduced.
- Production checkpoints and GPU/worker/process checks are preserved as
  `production_checkpoint_l1.json`, `production_checkpoint_l2.json`,
  `production_gpu_l1.json`, `production_gpu_l2.json`. New jobs399733/399747
  retain batch64,500epoch budget, original model/EMA/optimizer/scheduler/RNG
  continuation, per-run locks and48h requeue-enabled combined partitions.
  S1--S4 and the original Transformer were not changed. Measured gains apply
  to these short post-warmup windows; they do not guarantee constant whole-run
  throughput or bitwise equivalence after the explicitly recorded CUDA-flag
  transition. No convergence or simulator-result claim is made.
- Final L2 writer-node checkpoint read verifies step24000/EMA24000,
  epoch21/batch606, model/EMA/optimizer and all five RNG categories; provenance
  remains unchanged. Live training had advanced to24520with finite losses and
  gradients. Final queue check contains exactly the two intended LC runs,
  both RUNNING, with all protected jobs still RUNNING and no obsolete child
  queued. Diff checks pass. Authorized test-and-continuation work is complete.
