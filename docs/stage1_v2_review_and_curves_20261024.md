# Stage1 v2 four-H200 review and single-H200 curves

## Scope and safety

- Review date: October 24, 2026, Asia/Shanghai. This is an independent
  throughput experiment, not permission to restart or migrate production.
- Production384201/server09 and checkpoint-retention385697/server05 remain
  running. All six previously recorded production source/config hashes match
  again in `production_hashes_review_20261024.json`. No production checkpoint
  is loaded for training or overwritten by this review.
- Existing GPU job386479 is preserved, not cancelled or resubmitted. It keeps
  fourH200,64CPU,768GiB, full4days and server09 exclusion. The immutable
  `gpu_source_v3/` hashes were verified again; no benchmark source was changed
  in this review. At18:43CST it remains PENDING/Priority.

Experiment root:
`/data/wangzihao/outputs/manifeel/manifeel_stage1_v2_ddp4_async_global128_20261024_132711`.

## Reviewed paths and evidence

| Area | Review / existing gate | Verified now vs still outstanding |
|---|---|---|
| Previous quick exit | Four-thread Normalizer fit, exact preserved hash, restore serial parent before fork even on errors | Real400-demo tensors match; no tolerance relaxation. CPU startup now passes the exact previously failing path. |
| Data identity | Original source/cache contracts, full-array SHA256-derived block index, per-block verification before publication | All63arrays/4532blocks indexed previously;32real balanced windows checked bitwise again, including history/future masks. |
| Cold startup | One process-shared RAM reservation, demand-filled verified blocks; background full-cache fill only after first real update | CPU startup reserves299494742868bytes and verifies214blocks for its real-data checks. Actual CUDA startup latency remains unmeasured. |
| Fork safety | CUDA-free serial parent; fork rank-local DataLoader workers before setting CUDA device or initializing NCCL | Exact parent state and launch signature pass CPU review. Real CUDA/NCCL initialization still requires the allocation. |
| Batch and sampler |32/rank,global128, original balanced continuous stream, no tail dropping, cursor based on completed warmup+timed updates | CPU shard identity/resume regressions pass. GPU run audits initial identities, nine-task counts and the exact next checkpoint batch. |
| Input overlap | Worker assembly, bounded pinning thread, dedicated H2D stream, batch-specific wait event, record_stream and retained pinned buffers | Scheduling/lifetime CPU regressions pass. Real overlap and speed remain unproven. N+1 copy is enqueued after backward and before the clipping finite-check CPU synchronization. |
| Loss and precision | Same policy, weighted objectives, dropout sampler,FP32, clipping, optimizer and EMA order | Production-bound policy/schema sources match; real392799231-parameter policy and optimizer construct successfully on CPU. Constant LR is explicitly benchmark-only. |
| Gradient and EMA state | DDP mean for equal32-sample shards; frozen target/Normalizer; whole-model replica audits and EMA formula check | Four-process CPU Gloo gradient regression passes. NCCL replica and real EMA checks are mandatory gates in the GPU preflight. |
| Checkpoint | Rank0-only atomic independent file, config+Normalizer+optimizer+scheduler+per-rank RNG+sampler cursor | CPU replay regression passes. Real GPU next-update bitwise replay runs outside the timed window and must pass. |
| Measurements | Synchronize only at stable-window boundaries; CUDA events for H2D/compute/input waits; exclude warmup, verification, audits and checkpoint I/O |2000stable steps per case; matched CPU threads. No hardware throughput, CUDA overlap or utilization result exists yet. |
| Production isolation | Separate frozen source, independent output roots, no production resume or checkpoint path writes | Production remains running; original hashes unchanged. |

### Exact-entrypoint CPU startup review

- Submitted CPU-only386506/server44 with16CPU/768GiB and no GPU. Completed
  successfully in3m13s, exit0:0. The allocation reserves RAM but does not need
  to fill the entire cache for these startup checks.
- Imported the exact frozen queued GPU entrypoint, executed its real dataset,
  Normalizer and32-window validation, then intercepted only the GPU monitoring
  process and rank-launch boundary. Constructed the real policy and optimizer
  on CPU, checked frozen EMA/Normalizer and exact global128 shard identities.
- Result: `cpu_review_386506/review.json`, `passed=true`, CUDA uninitialized,
  parent Torch threads1, Normalizer hash
  `7dc801fb6c9b2edfed3a365738e838a47d6e5c72d0d5903ad30835e78f2ea16f`.
- Added `afterok:386506` to the existing queued GPU job while reviewing it.
  Successful completion cleared the dependency without losing the original
  submission or creating a duplicate GPU job.
- This is not a four-rank CUDA execution or a performance result. It prevents
  repeating the already diagnosed CPU-startup failure; it cannot establish
  NCCL, CUDA memory, copy overlap or GPU checkpoint determinism.

### GPU gates retained

1. Log assigned GPU model/memory/utilization/processes/topology at startup.
2. Run24stable-step four-GPU copy-stream and pinned preflights, including
   synchronization, EMA, sampling and checkpoint audits. Verify initial losses
   and gradient norms match across the two input pipelines within1e-6.
3. Only then run2000stable steps each for four-GPU copy-stream, four-GPU pinned
   and matched single-GPU copy-stream. Start timing only after all blocks are
   verified/resident and minimum warmup has completed.
4. Compare against both historical39.7samples/s and the matched single-GPU
   result. Global128 throughput does not establish unchanged convergence.

CPU verification in this review:19targeted tests pass, including the two new
plot-reading regressions; shell syntax and immutable source hashes pass.
Full suite:283passed,1skipped,1known unrelated failure in154.76s:
`tests/test_manifeel_eval_config.py::test_film_checkpoint_uses_local_assets`
cannot find its local CLIP source. Log: `full_tests_review_20261024.log`.

## Single-H200 training curves

Read-only snapshot: October24 **18:31:48CST**, step96480, epoch26 in progress,
25completed epochs. The trainer continues independently after this snapshot.

Artifacts:
`/data/wangzihao/outputs/manifeel/analysis/stage1_v2_single_loss_20261024_183149`.

- `training_metrics.png`: all six metrics; `training_metrics_recent.png`: last
  20000steps; `training_metrics.pdf`: full-resolution overview.
- `progress_snapshot.jsonl`, `metrics.csv`, `summary.json`: exact reproducible
  snapshot and values. Snapshot SHA256:
  `e289cbc52804070bc8d7358652bf4d795dac2ca08681d4086d683bfb5f7e72fc`.
- Gray lines are logged batches, the solid line is the trailing mean of50logged
  observations, and dots are full completed-epoch sample-weighted means.
  The smoothed values are not averages over every unlogged training step.
- Four loss panels use a log scale; two variance panels use a linear scale.
  The future losses are unweighted. Total is
  `action + 0.5168797374060089 * future_vision + 0.4334887418589181 * future_tactile`.
  Epoch totals are reconstructed because epoch records do not log total loss.
- These are the original mixed-input/dropout training metrics, not a new
  full-input-only evaluation.

| Metric | Latest50logged-batch mean,steps95500–96480 | Full epoch25 mean |
|---|---:|---:|
| Total loss |0.03519412|0.03564244|
| Action loss |0.01940798|0.01984377|
| Future vision loss |0.01720920|0.01725075|
| Future tactile loss |0.01589674|0.01587609|
| Vision latent variance |0.07956361|0.08139933|
| Tactile latent variance |0.07544522|0.07464979|

The variance metrics are **EMA future-target** latent variances, computed as
`target.var(dim=(0, 1), unbiased=False).mean()` across batch/future-time and
then averaged across view/features. They are not variances of the online fused
state token. Both have declined substantially from their earlier peaks; neither
is currently zero, but these curves alone cannot rule out representation
collapse or establish that reduced future loss means improved prediction.

The six-panel PNG was visually inspected. The inline six-facet rendering passed
JavaScript execution with a mock DOM at320/736px, finite-geometry checks and
epoch/metric completeness checks; no real-browser rendering was available.
