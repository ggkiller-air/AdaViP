# Independent Stage2 infrastructure benchmarks

- User authorized separate performance tests and official-document research,
  explicitly protecting all existing running/queued production jobs. No
  production process affinity, precision, checkpoint or allocation was changed.
- Official Slurm sbatch/srun docs confirm one job can target `gpu,gpux`, and
  explicit step CPU requests avoid a one-CPU default/inherited affinity.
  The installed22.05cluster accepts the combined request in both test-only
  validation and actual benchmark submissions.
- Four future submit wrappers now share `submit_partition_job.py`: one job ID,
  queried current partition limits, common maximum48hours (gpu100h/gpux48h),
  requeue eligibility, preserved dependency/resume exports, per-run lock and
  one-ID queue files. Updated AGENTS scheduling preference as requested. Existing
  immutable snapshots retain their original submission behavior.
- Nine focused regression tests pass in6.26seconds, covering all four wrappers,
  exactly one sbatch invocation, partition/time/requeue/dependency options and
  continuation paths. CLI help/import, Python compilation, shell syntax and
  diff checks pass. Default pytest FD capture again encountered the known
  filesystem truncate error; `--capture=sys` succeeds.
- Added a bounded real-shape/real-data benchmark for S2 and S3 (S4 shares S3's
  architecture): actual batch32/T2/U-Net, representative C-stage optimizer rates,
 72RAM-resident K64 windows balanced8per task,10warmup/80timed updates per case.
  Twelve cases isolate CPU affinity, worker8/4/0, deferred CPU metrics, foreach
  EMA, GPU-resident upper bound, TF32/cuDNN, channels-last weights, BF16, fused
  AdamW and U-Net compile. Preserves the original per-step losses/clipping/EMA
  except explicitly identified metric/EMA alternatives. CUDA-synchronized wall
  timing excludes separate profiler capture and reports cold warmup separately.
- Research sources: Slurm sbatch/srun, PyTorch performance tuning and AdamW,
  NVIDIA deep-learning performance guide. Method, sources and interpretation
  limits are in `docs/manifeel_infra_benchmark.md`. Installed Torch2.4.1 supports
  fused AdamW and foreach EMA. No package upgrade or protocol change was made.
- Final isolated immutable source:604SHA256-inventoried files under
  `/data/wangzihao/outputs/manifeel/stage2_infra_benchmark_20261029_120115/source`.
  Independent benchmark jobs S2/399130 and S3/399131 use a single combined
  partition request each, normal `yangyc/user_wangzihao`,oneH200/32CPU/64GiB,
  one-hour bounded runtime and requeue eligibility. Atomic per-case reports
  support restart by repeating only unfinished measurements. Separate monitor,
  stdout/stderr and optional profiler trace remain in each output sibling.
- Replaced only this investigation's still-pending draft benchmark jobs399118/
 399119 before finalizing the measurement matrix. Original L1/L2/S1 and S2--S4
  requests were untouched. S1/399040 independently started on server23 during
  this investigation; S2--S4 and the existing Transformer continue running.
- At the final check both benchmark jobs remain PENDING/Priority. No H200
  speedup, precision-equivalence result or completed profiler run is claimed.
  Actual GPU measurements await independent allocations; CPU code checks are
  not substituted for those results.

## Startup repair at12:42--12:46

- Both jobs obtained H200/server46 allocations but failed before measurements:
 399130 at12:13:45--12:14:09 and399131 at12:14:46--12:15:00. Their64GiB subset
  benchmark was rejected by the reused full299GB-cache RAM-capacity guard.
  No case completed. Allocation startup correctly exposed32logicalCPUs, unlike
  the production trainer's two-thread binding.
- Added explicit `require_full_ram=False` for benchmark metadata/subset mapping.
  Default/full-loader capacity checks remain enabled; regression verifies both
  rejection for insufficient full-cache memory and successful subset validation.
- Added CPU-only `--check-inputs` exercising the actual benchmark data path.
  At a64GiB allocation setting it successfully prepares72real windows,8/task,
  with expectedT2/vision/tactile/state/language/action shapes. This validates
  startup inputs only, not GPU performance. Also verified actual two-worker
  tensor transfer using a persistent short socket path.
- Related checks:13tests pass in44.71seconds; the new step-wrapper regression
  passes in24.28seconds. Evidence under
  `/public/home/wangzihao/.cache/lc/infra_fix_20261029/` and copied into retry root.
- Retried independent benchmarks as399169(S2)/399170(S3), each still one
  gpu,gpux job. Frozen source and reports under
  `/data/wangzihao/outputs/manifeel/stage2_infra_benchmark_retry_20261029_124611/`.
  The retry uses persistent short TMPDIR plus real IPC validation inside srun.
  Both are PENDING at12:46; no speedup result is claimed. No running production
  job was adjusted or restarted.

## First completed measurements,13:11

- Both retries completed successfully:399169(S2)12:48:59--12:57:17,
 399170(S3)12:49:59--12:57:44. All12cases per strategy produced finite loss/
  gradient measurements, with reports, profiler traces and operator tables.
  Compact raw-result mapping is `summary.json` in the retry root.
- Samples/s, S2/S3 respectively: observed two-thread baseline141.39/143.88;
 32logicalCPU+8workers142.48/165.70;32CPU+4workers154.79/148.45;
  zero workers85.22/103.64; deferred metrics88.98/146.34;
  foreach EMA110.20/152.68; GPU-resident157.10/148.42;
  TF32/autotune100.29/143.99; TF32/channels-last95.34/134.52;
  BF16/channels-last145.68/119.40; BF16/fused117.85/119.66;
  compiled U-Net/BF1694.03/124.37.
- CPU allocation changes yielded only0.8%(S2) and15.2%(S3) with8workers in this
  single small-pool trial. The affinity defect is verified, but earlier suspicion
  that it explains most slow training is not established. Precision/compiler
  alternatives show no consistent large gain and are not adopted in production.
  These ordered80-step trials on72windows do not establish stable full-dataset
  speedups; randomized repeats and representative whole-cache measurements are
  needed before attributing differences or recommending rollout.
- Profile hotspots include convolutions, GroupNorm, AdamW and layout conversions.
  GPU-resident input is not uniformly faster than the corrected-CPU baseline,
  another reason to avoid treating this single sweep as a precise ranking.
- Production read-only check: S1/S2/S3/S4 at epochs39/77/73/76; L2 at epoch4,
  approximately103.5samples/s. L1/399171 passed actual persistent-directory IPC
  validation and saved step3before cache loading. At13:11 only state/action
  completion events were emitted; server51 parentRSS was approximately27.3GiB
  and read_bytes approximately19.5GiB, so cache loading remained incomplete.
  No new stderr error. All running jobs remain untouched.
