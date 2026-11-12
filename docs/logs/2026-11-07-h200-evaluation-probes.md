# Single-GPU H200 evaluation feasibility queue

- User requested testing current benchmark checkpoints on H200, explicitly
  allowing overload idle GPUs, no fixed nodes and no large parallel allocation.
  Read `/public/home/wangzihao/AGENTS.md`: historical ManiFeel camera behavior
  varies by physical GPU; full H200 rollouts have succeeded. Require the exact
  assigned GPU to pass bounded camera diagnostics before checkpoint rollout.
- Submitted/released job **436508**, account/QOS overload, one H200, 16 CPUs,
  96 GiB, gpu,gpux, current common maximum 48h, Requeue=1. Both requested and
  excluded node lists are null. Only this new GPU job exists. Existing UniVTAC
  training426602, progress423354 and interactive425310 were left untouched.
- Persistent scripts/configuration/receipts:
  `/data/wangzihao/outputs/h200_eval_20261107_0135/`. Source archives preserve
  AdaViP1a118dd and UniVTACd6ea66f; existing ManiFeel third-party packages are
  linked read-only. GPU/UUID/process telemetry precedes work. Keep the prior
  independent16%memory reservation; all probes run serially in the allocation.
- GPU color and depth camera tests: 10environments,100frames,180s each,
  logical cuda/graphics0, require diagnostic_complete. Run real Isaac Sim4.5
  and5.0 RTX camera startup/render tests with360s limits, plus a Vulkan device
  extension report. Failed/partial/timed-out cases never produce success scores.
- Actual recorded-observation inference checks: UniVTAC DP retained E300 on
  its six covered tasks; Tabero DP E200; six FTP Students step20000 EMA;
  Tabero-VTLA step50000 raw. Restore actual weights and require finite action
  outputs of their native shapes. These are not closed-loop success metrics.
- On camera-compatible assignments, run ManiFeel E600 first, then
  E20/50/100/200/300/400/500 across all nine tasks,10environments/500steps,
  EMA/full sensors, original frozen task embeddings and test/mean_score.
  Each task has1200s timeout; completed receipts persist across reallocations.
  Camera/rollout failure requeues the same job, at most six serial allocations
  including preemptions; no node affinity or exclusions. Native inference
  successes are retained. Six failures end the job with diagnostic receipts.
- Added isolated simulator environments under
  `/public/home/wangzihao/.venvs/uvip-isaacsim-probe-{4.5.0.0,5.0.0.0}`,
  complete default-app and offline extension caches.4.5 uses Torch2.5.1/
  NumPy1.26.4;5.0 has its own upstream-pinned Torch2.7.0/NumPy1.26.0.
  No training environment was modified. Isaac Lab/TacEx benchmark runtime
  setup remains separate; successful simulator camera probes alone do not
  establish complete UniVTAC/Tabero benchmark rollouts.
- Validation before release: Python compilation and shell syntax passed;
  harness checks verified successful marker, missing-marker rejection and
  subprocess timeout cleanup. ManiFeel E600 strict CPU EMA/normalizer load
  passed as Stage2L1Policy,314236552parameters. This is not a GPU test.
- At2026-11-07 01:52:59+0800 job436508 was released and PENDING/Priority;
  no allocated GPU or new rollout result was available at this snapshot.

## Morning audit and repair

- Job436508 exhausted six serial allocations and ended03:04:23, exit76.
  These were only two physical H200s, both server13:GPU4 twice
  (GPU-27641169-320d-70ac-c5b2-3065bcc05100),GPU2 four times
  (GPU-479e80e2-f92e-b6ca-0bb2-950d19f77d1a). No closed-loop score exists.
- All six FTPstep20000 EMA recorded-input inference cases passed with finite
  actions[1,32,120]; Tabero-VTLAstep50000 raw passed[1,50,32].
- Both IsaacSim4.5 and5 camera probes failed with Vulkan ERROR_DEVICE_LOST
  and native crashes. This is evidence for these assignments only.
- ManiFeel never reached camera startup: its editable IsaacGym install still
  pointed to the pre-reorganization AdaViP directory. Restored the Python path
  explicitly in the retry job to the existing licensed package under
  project-archive/2026-11-02/legacy/AdaViP/third_party/
  IsaacGym_Preview_TacSL_Package/isaacgym/python. Native gymapi import verified.
  No training environment installation was changed.
- UniVTAC DP E300 and Tabero DP E200 failed because CPU Tensor scheduler
  timesteps reached CUDA time-embedding layers. Both owned U-Net modules now
  move timesteps to sample.device; checkpoint topology/weights are unchanged.
  Added scalar/integer/batch timestep forward regression cases, including CUDA.
- Retry artifacts use /data/wangzihao/outputs/h200_eval_20261107_fix1/.
  Original failed-run snapshots remain intact; passed inference receipts are
  copied with provenance and skipped. Regression tests run on the allocated
  GPU before camera probes. Python import errors now stop instead of requeue.
- Submitted retry job437804 at10:31:02, running10:31:03. Same single-GPU
  overload configuration, no node request/exclusion. Assigned server13 GPU3,
  UUID GPU-da8386db-3d21-340e-c1a6-5013a7f04df9; initial0MiB/0%utilization.
- Both U-Net regression suites passed on allocated H200 GPU3 (six cases
  per suite including CUDA); corrected ManiFeel color camera preflight started.

## Six independent GPU tasks

- User explicitly changed concurrency to six tasks/six GPUs so ManiFeel camera
  stalls cannot delay other benchmarks. Cancelled serial job437804 after
  disabling requeue; its repaired camera probe reached all10camera creations
  and stalled at gpu_acquire_begin env=0, bounded timeout180s. Logs retained.
- Submitted array437836, indices0–5, concurrency6, each oneH200/16CPUs/96GiB,
  overload account/QOS, gpu,gpux48h, no requested/excluded nodes/dependencies.
  Indices:0ManiFeel,1UniVTAC DP E300,2Tabero DP E200/20/50/100/150,
  3FTP/VTLA recorded-input inference,4IsaacSim4.5,5IsaacSim5.0.
- The native lane keeps the six passed FTP20k and VTLA50k receipts with source
  provenance, then checks VTLA5k/10k/20k/30k/40k. These are inference checks;
  simulator lanes are camera feasibility probes, not full benchmark rollouts.
- Per-lane locks, attempt directories, memory holders, torch extension caches,
  timeouts and bounded independent requeues. Camera failure affects only its
  own array element. Passed checkpoint receipts persist. Inference report
  writes are atomic. Original source snapshot plus validated DP fixes reused.
- Shell syntax and Python compilation passed; harness verified six independent
  dispatch routes and that failed ManiFeel camera gates prevent rollout.
- Artifacts: /data/wangzihao/outputs/h200_eval_20261107_parallel6/.

## 11:13 result audit

- UniVTAC DP E300 inference passed all six recorded task samples. Tabero DP
  E20/50/100/150/200 and VTLA5k/10k/20k/30k/40k/50k all passed recorded-input
  inference. Six FTP20k results retain the earlier successful evidence.
  Array elements1/2/3 completed successfully; these are not success rates.
- ManiFeel passed both color/depth100-frame camera preflights on server13
  physical GPUs3 and4. E600 power-plug actual10-env/500-step rollouts started
  on both; each last logged64/500 then exited-11 (native segfault), not timeout.
  No complete rollout or mean_score. Later camera tests timed out, including
  repeats on the same GPUs: a previous pass does not guarantee stability.
- Both IsaacSim versions exhausted six attempts and stopped. Final logs show
  Vulkan ERROR_DEVICE_LOST/native crash; tested assignments all server13.
- Only ManiFeel437836_0 remains queued for its sixth/final allocation retry
  (BeginTime11:14:27). Status snapshot: parallel6/status_1113.json.
