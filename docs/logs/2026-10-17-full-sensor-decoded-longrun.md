# Lossless decoded cache and full-sensor long training

- User requested fixing throughput before continuing full training, recalling
  a previously decoded-data path. Stage0's new loader had no such cache and
  repeatedly decoded compressed Zarr chunks for random two-frame windows.
  Confirmed the remembered upstream path: ManifeelDataset calls
  ReplayBuffer.copy_from_path; withstore=None, copy_from_store materializes
  eachselectedarray into NumPy RAM. Stage0 had instead openedzarr groups
  directly. Read reference sources only; no third_party changes.
- Inspectedjob368013 allocation: H200143771MiB total/28109MiB used/0% sampled
  utilization; sole allocated computePID981918. SentUSR1 only to that trainer
  through an overlapping Slurm step. It savedlatest+interrupt atstep147 and
  exitedCOMPLETED0:0, preserving optimizer/EMA/RNG/cursor. Left366142 untouched.
- Added independent float32 decoded NPY cache for selected train+validation
  episodes, source-index mapping, deterministic resize, SHA256/contract checks,
  atomic completion and resumable per-array construction. RAM loader verifies
  once, shares read-only data with Linux fork workers; no quantization or
  scientific-protocol change. Current selected71,265 frames total278.93GiB.
- Resume now allows only lossless cache/loader transport/runtime-gate changes;
  old full-config hashes validated against checkpoint config before migration.
  Batch, split, normalizer, masks, optimizer and schedule remain strict.
- Initial focused cache/legacy-resume regression:4passed,35deselected,30.10s.
  Verified all train/val windows, future targets, normalizer and raw/EMA weights
  remain bitwise equal after replacing direct reads and changing worker count.
  Log `/data/wangzihao/outputs/manifeel/slurm/full_sensor_decode_focused_20261017.log`.
- SubmittedCPU cachejob368028,24CPU/64GiB/2h onserver21,16 decode threads,
  block48, persistentcache `/data/wangzihao/cache/manifeel/full_sensor_float32_9task_v1_seed42_20261017`.
  Frozen builder/config/hash under
  `/data/wangzihao/outputs/manifeel/full_sensor_cache_build_20261017_134757`.
  At13:50:05 RUNNING1min41s, no stderr; not yet a complete manifest.
- Prepared separate1H200/24CPU/384GiB/48h launcher, samebatch64/300epochs and
  full9-task validation. First5x20-step windows gate stable throughput>=150/s
  and datawait<=35%, ignoring warmup. Failed gate saves+exits nonzero rather
  than wasting the allocation; passing gate continues in-process. Only a
  time-limitSIGUSR1 safely resubmits same-source continuation if incomplete.
  Full regression and actual GPU throughput validation still pending.
- Expanded regression (RAM/mmap, resize, corrupt cache, memory guard, legacy
  checkpoint migration with worker-count change, throughput gate):48passed
  in50.33s. Full repository:182passed,1skipped,1known unrelated FiLM CLIP
  source-asset failure in66.78s. Logs `full_sensor_decode_all_20261017.log`
  and `full_sensor_decode_repo_20261017.log` under the persistent Slurm logroot.
  Shell syntax and `git diff --check` passed.
- At13:52:16 submittedlongjob368047 with afterok:368028, sameyangyc/gpu H200
  resource route as approved Stage0 runs; no GPU occupied while cache builds.
  Frozen source/hash/Git metadata and new output directory:
  `/data/wangzihao/outputs/manifeel/full_sensor_reference_batch64_ep300_seed42_decoded_20261017_135210`.
  Resume source is368013's step147latest; independently compared resolved
  old/new scientific protocols, exact match SHA256
  `ca774c8c328857e89ee5bef0087ec7bed8329a3c1166111bf07a148ca4f83935`.
  Pending dependency at submission; real GPU throughput remains to be checked.
- CPUjob368028 completed0:0 in13min18s; decoder reported779.02s,63/63 arrays,
  299,494,742,868 bytes (278.93GiB), atomic complete manifest published. Final
  MaxRSS47,824,244KiB. No stderr. Dependency released; at14:04:44 longjob368047
  was stillPENDING/Priority, so no actual GPU throughput claim yet.
- Real-data check (read-only maps, no full extra RAM copy): all9 tasks,
  36first/last train+validation episode-boundary windows bitwise equal in
  observations/actions/masks/indices. Train69,205/validation2,060 unchanged;
  direct and cached train-only normalizer identical SHA256
  `6cf5e6e360081271525e87f5a6024364c449835425490ac3a522301e177351c4`.
  Log `full_sensor_decoded_real_equivalence_20261017.log` in persistent logroot.
  Frozen training source SHA256 verification also passed.
- Longjob368047 initially waited for priority. SetTimeMin6h without reducing
  its scientific epoch budget; scheduler backfilled6h onserver37 at14:08:28.
  Allocated H200143771MiB was empty/no compute apps; IPC59-byte check passed,
  /dev/shm521GiB free. Serial cold load read about16GB in92s, only~170MB/s;
  this would repeatedly waste substantial GPU time after allocation changes.
- Cancelled only368047 during RAM load, beforeworkspace_ready or any new
  optimizer update. Its oldstep147 checkpoint remains untouched. Added8-way
  parallel verified RAM preload and persistedmin6h/max48h backfill settings in
  launcher so time-limit continuations can also backfill. Post-change related
  regression48passed48.78s; log `full_sensor_parallel_load_20261017.log`.
  New frozen snapshot/submission replaces the cancelled serial-load run.
- Replacementjob368109 submitted14:12:25, RUNNING onserver51. New frozen
  source/output:
  `/data/wangzihao/outputs/manifeel/full_sensor_reference_batch64_ep300_seed42_decoded_parallel_20261017_141224`.
  Allocation preflight: H200143771MiB empty/no compute apps; /dev/shm319GiB
  free;59-byte IPC socket/two-worker tensor transfer passed. At14:17:58
  elapsed5min29s, parallel preload had read258.5GB (~0.79GB/s including startup),
  versus~0.17GB/s serial cold load. MaxRSS261,431,324KiB; stderr empty.
  No optimizer-throughput claim until loading/provenance restore and gate finish.
- Parallel cache load completed in351.81s (all299.49GB verified). Actual GPU
  workspace restoredstep147 with original manifest and normalizer hashes
  unchanged (`936a0ae8...`/`81ff7294...`, OMP/MKL4 as original launcher).
  Separate real-data equivalence script usedOMP/MKL1; its `6cf5e6e3...`
  fingerprint is direct-versus-cache equality under that thread setting, not
  a replacement for the original training normalizer. No protocol bypass.
- At14:19:58 step240 runtime gate PASSED: four post-warmup20-step windows,
  harmonic throughput247.56samples/s, datawait0.197% (old~21–27/s/~92%).
  Same process continued full300-epoch training. By14:20:49 step440, latest
  throughput274.99/s, datawait0.176%, loss0.09688, gradnorm1.6413 finite;
  Torch peak24.78GiB. GPU monitor29875MiB occupied,93% utilization at14:20:44.
  Actual first allocation6h through20:12:29, signal warning5min beforeend;
  launcher includes automatic same-source/same-protocol continuation onUSR1.
  First new-run checkpoint/full validation still to be verified.
- First complete EMA validation passed on all9 tasks/2060 held-out samples:
  epoch1/global1082 full macro loss0.04289798, finite ten-condition metrics.
  Newlatest andbest saved, each5,008,126,966bytes. Training enteredepoch2.
- Worker exit emitted nonfatal NAS `.nfs* Device or resource busy` tempdir
  cleanup traces; optimizer/validation/checkpoints continued successfully.
  Verified node-local `/dev/shm/avip_probe_368109` IPC and repeated8/4/8/4
  worker lifecycles without NFS errors (probe's1CPU affinity only caused the
  expected worker-count advisory). Logs `full_sensor_local_ipc_probe_20261017`.
- Fixed decoded launcher TMPDIR to short node-local `/dev/shm/avip_JOBID`;
  persistent cache/logs/checkpoints remain under/data. RequestedSIGTERM only
  for trainerPID2495722, preventing old-launcher USR1 auto-submission. Saved
  exactepoch2 batch632/global1714 and exitedCOMPLETED0:0; no step rollback.
- Submitted continuationjob368155 at14:27:54 afterok:368109, same scientific
  protocol and full target, new frozen source/output:
  `/data/wangzihao/outputs/manifeel/full_sensor_reference_batch64_ep300_seed42_decoded_localipc_20261017_142753`.
  Preserved parentbest via hardlink (future saves atomically replace only the
  new path), copied prior progress/epoch1 metrics, and resumes parent's final
  latest with optimizer/EMA/RNG/cursor. Shell syntax/diff checks passed; no
  Python training/model change since the48-passed parallel-load regression.
- Finaljob368155 RUNNING onserver36 since14:30:34. Preflight found an empty
  H200143771MiB/no compute apps and168GiB free/dev/shm. Local52-byte IPC/tensor
  check passed. Cold RAM preload603.71s on this node (NAS rates vary), no
  repeated Zarr decoding. Restoredexactepoch2 batch632/global1714 with same
  manifest/normalizer/protocol hashes and preservedbest/history.
- New-job post-warmup gate PASSED atglobal1800:252.31samples/s,0.193% data
  wait. At14:47:13 global2800/epoch3, recent245.22/s,0.193% wait, finite
  loss0.03159/gradnorm0.2391; Torchpeak24.78GiB. GPU monitor at14:45:52:
  29885MiB occupied,96% utilization. User's CPUjob366142 remained untouched.
- Verified on writer node through overlappingSlurm CPUstep: full epoch2
  EMA validation covered9tasks, full macro loss0.03552967 (epoch1:0.04289798),
  trainloss0.04652234, trainseconds296.14, effective233.69samples/s excluding
  validation/checkpoint/startup. latest+best each5,008,126,966bytes, independent
  inodes after improvement; stderr0bytes after full-size worker shutdown,
  validation and next-epoch workers. Thus local IPC fix verified, not just
  synthetic probe success. A cross-node NAS metrics read briefly showed only
  epoch1; writer-node read confirmed both records (likely cache/coherence delay).
- Current allocation ends20:30:34, withUSR1 warning at20:25:34. Launcher
  automatically submits same-source continuation only after safeUSR1 stop,
  until300epochs; ordinary cancellation/failure does not auto-restart. Pure
  train-time extrapolation fromepoch2 is~24.7h for300epochs; validation,
  checkpoint I/O, cold-start and queue delays are additional, not guaranteed.
  Final sourceSHA256 and whitespace checks passed. No third_party edits,
  no branch/commit/push. Relevant tests48passed; prior full suite retains
  only its known unrelated missing-FiLM-CLIP-asset failure.

## Evening continuation/checkpoint audit, 2026-10-17

- At21:28 confirmed original368155 completed0:0 at20:25:42 after safe
  SIGUSR1 stop (step72509, completedepoch67,next epoch68/cursor15). Existing
  launcher automatically submitted369207; it started21:13:50 onserver28,
  resumed the exactstep72509/cursor15 and passed throughput preflight225.36/s.
  Queue gap48min08s occurred; therefore this is automatic resumable training,
  not a promise of one uninterrupted process until300epochs.
- Current369207 isRUNNING1H200/24CPU/384GiB; sampled H200143771MiB,
  29885MiB used/84% utilization, real optimizerstep74080 inepoch69. Current
  allocation ends2026-10-18 03:13:50, USR1 warning expected03:08:50, same
  safe-save/afterok continuation mechanism. Frozen config still300epochs,
  every20epoch snapshots, latest each1000updates and epoch, best on full EMA
  macro validation. No1000epoch or100epoch-retention migration has occurred.
- Six complete .ckpt files currentlyabout4.66GiB each/28GiB total:
  latest,best,interrupt,epoch20,epoch40,epoch60; no.partial files observed.
  Actual CPU deserialization confirmed lateststep74000/EMA74000,
  completedepoch68/cursor424, and interruptstep72509/EMA72509,
  completedepoch67/cursor15. Both contain raw/EMA/optimizer,482Adam states,
  Python/NumPy/Torch/CUDA RNG and300epoch config, not weights-only files.
  Snapshot files are atomic .partial→.ckpt replacements. Checkpoint file mtimes
  differ from Slurm/log event times; use embedded counters and Slurm timestamps
  rather than inferring training progress from NAS mtimes alone.
- Latest completed validationepoch68 full EMA macro0.03700587, nine tasks;
  best remainsepoch18/step19476 full EMA macro0.02787355. Validation has not
  monotonically improved; best retention works and latest is not advertised
  as the best model. Ordinary training errors/OOM/node failure are not covered
  by the launcher's SIGUSR1-only automatic continuation, so no unconditional
  guarantee of uninterrupted progress to300epochs is possible. No jobs
  stopped, no training config changed for this read-only audit.

## Stop requesting six-hour backfill slices

- User objects to repeated continuation instead of one sufficiently long
  allocation. At21:31 confirmedgpu partition permits100h; the six-hour limit
  was our --time-min=06:00:00 scheduling choice, not a cluster-wide cap.
  Attempted `scontrol update JobId=369207 TimeLimit=2-00:00:00`; scheduler
  returnedAccess/permission denied, so current allocation remains6h. Training
  remains running and was not cancelled to force a longer queue request.
- Removed --time-min from the repository single-GPU decoded launcher; future
  submissions from new snapshots request full48h. Do not edit the current
  immutable training snapshot to hide how the live job was launched.
- SubmittedCPU-only scheduling helper369423 (1CPU/512MiB/max5min), dependency
  afterok:369207. When current job safely exits/submits its original successor,
  helper identifies that run's recorded successor, verifies matching launcher
  andPENDING state, temporarily holds it, requestsTimeLimit=48h/TimeMin=48h,
  verifies, then releases. No GPU allocation and no concurrent trainer. If the
  successor already runs, parent failed or update is denied, fail without
  cancelling training; no assertion that the future update has already worked.
  Pending48h increase still needs actual scheduler acceptance when helper runs.
- Helper frozen underrun/scheduling withSHA256/job ID; does not mutate model,
  epoch300 budget, LR or existing checkpoint retention. Five mock-Slurm
  regressions pass in1.85s, covering normal update and rejecting unrelated,
  already-running or failed-parent jobs. Shell syntax/whitespace checks pass.
  Current target remains300epochs;48h should cover its remaining training at
  observed epoch duration, barring errors. This is distinct from the previously
  authorized but not yet implemented single-GPU1000epoch budget extension.
