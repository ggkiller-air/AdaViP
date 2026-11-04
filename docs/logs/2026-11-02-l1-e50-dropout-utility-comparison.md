# L1 E50 input-dropout utility comparison

- User selected **E50 for both teachers**, superseding the initial E200/E500
  proposal. Original checkpoint:
  `language_conditioning_t2_batch64_ep500_seed42_20261028_204459/l1/checkpoints/epoch50.ckpt`;
  dropout checkpoint:
  `l1_restored_batch64_ep500_seed42_20261101/l1_dropout10/checkpoints/epoch50.ckpt`.
  CPU restoration verified both E50/global_step55700/EMA and the identical
  saved normalizer `7dc801fb6c9b2edfed3a365738e838a47d6e5c72d0d5903ad30835e78f2ea16f`.
  No optimizer or normalizer fitting is involved.
- Yesterday's original E500 archive is complete (65,265 windows). Reuse its
  established legacy K64 reference streams and complete-window indexing for
  both new E50 stores, including pilot overlap; do not use its E500 utilities
  as the E50 comparison. Retain history2/horizon16, masked6D/7D actions,
  strict FP32/noTF32/noAMP, ten paired conditions and raw signed utility.
  Input dropout is disabled during scoring; interventions explicitly zero
  projected sensor+identity tokens and late language, with six presence flags.
- Extended teacher validation to reject wrong snapshot/EMA/step, training
  protocol or normalizer. Regression checks include all ten native dropout
  conditionings and rejection of mismatched comparison epochs/noise streams.
  Focused checks:37passed/1CUDA-skipped in35.94s. Initial pytest FD capture
  failed on an unlinked shared-filesystem tempfile before any tests ran;
  reran with `-s` and persistent redirected logs. Full regression is running
  with the previously documented hanging spawn-cache test deselected.
- Frozen722source files (including pinned Diffusion Policy), source inventory,
  checkpoint audit and independent label configs are under
  `/data/wangzihao/outputs/manifeel/l1_epoch50_dropout_comparison_k64_20261102_201244/`.
  Submitted **416813**, one independent H200/32CPU/384GiB, sequential teachers,
  gpu,gpux/common queried48h, requeue and root writer lock. Per-teacher atomic
  shards retain safe continuation. Existing five training jobs are untouched.
- Server30 has no CUDA device; before submission server13's visible NVIDIA
  H200143771MiB showed28725MiBused/4% and an existing Python compute process.
  New allocation inspects its own GPU/processes before scoring. At20:15CST
  job416813 is PENDING/Priority; no GPU preflight or E50 label completion is
  claimed yet.
- Automatic final comparison verifies completed stores, matching indices,
  epoch/step/EMA, normalizer, sampling and precision protocols. Outputs go to
  `original/`, `dropout10/`, and `comparison/` (pooled/per-task CSV, Markdown,
  PNG/PDF and COMPLETE.json). Report full-input loss alongside utility means,
  negative fractions and correlations; these are training-demonstration
  denoising diagnostics, not simulator rollout success. Overlapping windows
  are not independent observations.

## Allocated and initializing

- Job416813 started onserver38. Both allocation-local and independent SSH
  inspection show NVIDIAH200143771MiB,0MiBused/0% and no compute process
  before initialization. IPC transport passed with two actual workers.
- Original E50 scoring workspace has verified65,265windows, E50/55700/EMA,
  the expected normalizer and strict FP32 runtime. Full decoded RAM cache is
  loading. No completed GPU loss preflight or production shard is claimed yet.
- Full regression completed:355passed/2skipped/1deselected/5existing warnings
  in233.82s. The deselected test is
  `test_spawn_worker_reattaches_shared_decoded_maps`; no other test excluded.
  Persistent `focused_pytest.log` and `full_pytest.log` are saved in the run root.

## GPU preflight passed; production scoring active

- Original E50 loaded/verified all299494742868cache bytes in475.97s.
  Four real windows (tasks0/2/4/8),64draws and all10conditions passed the
  one-draw reference check:maximum absolute loss difference1.7881393432617188e-7.
  Persisted legacy draws match exactly, normalizer is unchanged and the
  teacher has0trainable parameters. Native full-input conditioning is exact.
- First256-window shard committed in33.86s,7.56windows/s; original-store
  remaining estimate2.39h, roughly5h for both teachers plus comparison at
  this early rate. No E50 final completion or comparison result is claimed.
  H200 utilization99% during preflight; trainer affinity contains the32
  allocated CPUs (52--67,148--163). GPU stderr is empty.
