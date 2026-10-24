# Stage1 v2: independent four-H200 asynchronous-input debugging

## Scope and protection

- Authorized on2026-10-24:4H200, global batch128, local batch32. Target roughly
  120samples/s;100+ is useful, but neither target nor3–4x scaling is guaranteed.
- Production384201/server09 and checkpoint archive385697 are not restarted,
  signalled, modified, migrated or used as benchmark output directories.
- Read-only production GPU snapshot: H200143771MiB total,40799MiB used,71%
  utilization. The Codex terminal is in a CPU allocation, not a selected GPU.
- Preserve model, FP32/TF32 settings, loss weights, original balanced task
  stream, optimizer definition, EMA and the all400-demo Normalizer. Global128
  is an explicit experimental protocol, not a claim of equal convergence.

## Input pipeline

1. CPU-only job386007 validates each complete original decoded array against
   its existing SHA256 and generates64MiB-row-aligned block hashes. This job
   requests no GPU and writes a new index, not modified cache files.
2. The GPU parent checks original source metadata/coverage, block-index binding,
   file identities and memory capacity. It reserves one demand-paged MAP_SHARED
   RAM allocation, shared across forked ranks and their CPU workers.
3. Required blocks are copied into RAM and verified before access. Readiness is
   published under process-shared locks; corrupted blocks are never accepted.
   Existing sample order is not changed to prefer cached samples.
4. After the first genuine optimizer update, rank0 starts background cache fill.
   All ranks keep training during cache warmup, rather than waiting for279GiB
   before initializing CUDA. Stable timing begins only after every cache block
   is verified/resident and the minimum number of warmup updates is complete.
5. Existing background pinning feeds an explicit copy-stream prefetcher. After
   backwardN is enqueued, but before gradient-clipping finite checks can
   synchronize the CPU, batchN+1 is staged on a rank-local CUDA stream. Compute
   waits on that batch's copy event, records tensor stream lifetimes, and host
   references remain alive until their H2D event finishes. No artificial GPU
   keepalive is used.

## Independent jobs and comparisons

- Experiment root:
  `/data/wangzihao/outputs/manifeel/manifeel_stage1_v2_ddp4_async_global128_20261024_132711`.
- Pointer:`/public/home/wangzihao/.cache/avs1v2/current_ddp_async.txt`.
- CPU source/hash is frozen separately from subsequent GPU benchmark changes.
- GPU launcher requests4H200/64CPU/768GiB/full4days, excludesserver09, and will
  depend on successful CPU index preparation. Initial GPU/process/topology
  queries are logged by the batch launcher before any benchmark starts.
- A24-step4GPU copy-stream/pinned preflight runs all existing correctness and
  exact-checkpoint-replay audits. Only a successful preflight permits stable
  runs:2000steps each for4GPU copy-stream,4GPU pinned,1GPU copy-stream.
- All cases explicitly use the same main-process CPU thread count. Do not
  repeat the previous1-vs8-thread confound. No per-step device-wide synchronize
  is inserted into either comparison; CUDA events measure H2D, compute spans
  and copy-stream input waits. Whole-window timing synchronizes at boundaries.
- Record samples/s,steps/s, per-rank CPU waits, H2D/events, peak GPU memory,
  NVML utilization, all task counts and DDP/EMA/Normalizer/checkpoint audits.
- Initial three losses/gradient norms must match across4GPU input pipelines
  within1e-6. Each pipeline uses the same or greater completed warmup count.
  Compare against both historical39.7 and matched1GPU; distinguish these ratios.

## Provenance and current validation

- Do not relax the production resume fingerprint or rewrite checkpoint
  provenance. The new benchmark initializes independent models; it does not
  resume a production workspace.
- The experimental source contract explicitly allows only the already recorded
  lossless `stage1_v2_dataset.py` change among production fingerprint-bound
  sources. Every other policy/schema/training/source-infrastructure file must
  match frozen production. Record both original and experimental source hashes.
- Before launching ranks, compare32 real balanced windows bitwise against the
  untouched production dataset implementation, including dictionary order and
  all action/history/future masks and identities.
-45related CPU regressions pass, including multiprocess shared readiness,
  corruption rejection, all-window dataset/Normalizer equivalence, unbounded
  sampler/resume order, mocked copy-stream lifetime/order and real4-process
  Gloo gradient synchronization. CPU scheduling mocks do not prove CUDA overlap.
- Latest full suite:280passed,1skipped,1known unrelated missing-local-CLIP-asset
  failure (`test_film_checkpoint_uses_local_assets`),121.46s. Log:
  `full_tests_v2.log` at the experiment root. Earlier v1 validation remains in
  `full_tests.log`. Shell syntax, Python compilation and diff whitespace pass.
- CPU386007 completed0:0 in8m16s:63arrays,4532blocks,278.926GiB, each original
  full-array SHA256 verified. No GPU was allocated during this preparation.
- GPU386023 is submitted, PENDING/Priority at13:45CST onOctober24. Scheduler
  currently predictsOctober26 08:00CST, not a promised start. Previous386012
  was cancelled only after filtering PENDING state; accounting confirms0runtime
  and no node assigned. It was replaced to correct the prefetch trigger before
  any GPU allocation, with the already verified CPU block index reused.
  No node pinning, shorter walltime or time-min was introduced; full4days,
  4H200/64CPU/768GiB and server09 exclusion are unchanged.
- Frozen pending GPU source is `gpu_source_v2/`, with
  `gpu_source_v2_sha256.txt` verified against every file. Earlier `gpu_source/`
  and its hash remain preserved, but that snapshot was never executed onGPU.
  Production source/config before/after hashes match exactly. No H200 CUDA
  overlap, four-GPU throughput or hardware correctness result exists yet.

## Quick-exit diagnosis and replacement,18:18CST

-386023 actually startedOctober24 18:05:45CST onserver44, but failed after20s
  before first optimizer update: Normalizer SHA256 mismatch. It was not OOM,
  an idle-policy kill or a wall-time expiry; all4H200 initially had0MiB used.
- Real400-demo inputs reproduce a thread-dependent mean reduction:1CPU thread
  differs from the preserved Normalizer only in action input_stats.mean by at
  most2.9802322387695312e-08.4/8threads exactly match every saved tensor and the
  original SHA256. Limits scale/offset do not differ; keep hash checks strict.
- Benchmark fitting now uses4threads and restores the serial parent's previous
  thread setting before fork, including error paths. Real-data validation in
  `normalizer_fix_real_data.json` matches original7dc801fb...6f and all pinned
  checkpoint Normalizer tensors bitwise.17focused regressions pass.
- Replacement386479 isPENDING/Priority at18:18CST, unchanged full4day request
  and4H200/64CPU/768GiB/excludeserver09. Frozen`gpu_source_v3/` and hash plus
  failed`gpu_source_v2/` remain separate. No production source/config change or
  checkpoint resume bypass. There are still no four-GPU speed measurements.
- Finalv3 validation:281passed/1skipped/1existing unrelated missingCLIP-source
  failure,134.02s (`full_tests_v3.log`);17focused regressions pass. Snapshotv3
  file hashes, shell syntax, Python compilation and whitespace checks pass.
