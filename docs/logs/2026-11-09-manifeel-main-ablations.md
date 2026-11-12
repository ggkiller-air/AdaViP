# ManiFeel L1 E50 main Stage2 ablations

- User approved B No Utility, C No FiLM, D No Utility-conditioned Query,
  E Neither, F Shuffled Utility. All five independently scheduled, no serial
  dependencies. Long jobs must use `yangyc/user_wangzihao`, never overload.
- Shared `Stage2L1Policy.ablation` constructs the original Full parameter
  sequence before pruning. B retains exactly the first 64 context columns of
  each FiLM input matrix; unused utility columns/predictor/projections are
  removed. C/E bypass FiLM exactly. D retains cross-attention and context query.
  Existing Full behavior, normalization, optimizer and global clipping remain.
- Actual Full initialization reproduced:
  `19c82812a5ac14d118e14337dc2bb8cda7b5071d0a1609c0cf8c2f1d6bf98803`.
  Every surviving tensor is checked exactly (B documented column slice), with
  no RNG consumption during pruning. Teacher is SHA95bd523e...fa5fb,
  L1 E50/55700 EMA; K64 SHA775a772d...bfe79. Both seeds42.
- Protocol clock: 2040 updates/epoch, A E1, B E2--6 (10200 updates),
  C E7--600 (1211760 updates). B/E skip A optimization but start protocol
  step2040/epoch2, preserving sampling seed42+epoch, cosine progress and C's
  500-update encoder ramp. Their E1 is an explicitly marked zero-update
  checkpoint; their actual optimizer count is protocol_step minus2040.
- Preserve E1/6/10/20/50/100/200/300/400/500/600 plus latest; archive first
  actual update independently. B/E have an empty utility optimizer group to
  preserve scheduler group ordering, with no Utility parameters or gradients.
- F fixed per-task whole-nine-vector bijection seed20261109, separately saved
  as utility_permutation.npy plus hash/task fixed-point counts. Original label
  files, RMS and observation/action alignment remain unchanged. Current Full
  trains all65265 complete windows; no held-out Stage2 validation exists.
- Gradient audit includes A where applicable, B/C, detached utility input and
  action conditioning, expected encoder freezes and active action modules.
  Parameter reports include each new head and actual gradient participation.
  Preserve global clip1; log weighted utility/action gradient norm means,
  clip coefficient and trigger frequency instead of claiming optimizer-level
  independence from detach alone.
- CPU regression:419passed,5skipped (227.97s); final main-ablation tests12passed
  (43.12s), including exact raw/EMA/Adam/RNG continuation, portable inference,
  unchanged Action weights/RNG through A, sampling and schedule boundary parity.
  Evidence: `/data/wangzihao/outputs/manifeel/main_ablation_development_20261109/`.
- Frozen source/registry/logs:
  `/data/wangzihao/experiments/manifeel/l1_e50_main_ablations_b32_ep600_seed42_20261109_2235/`.
  Initial overload smoke jobs451594--451598 were pending Priority with an
  estimated two-hour wait; cancelled only these newly submitted pending jobs
  and resubmitted short jobs under the user's normal account. Smoke B451621,
  C451622,D451623,E451624,F451625 all running concurrently. Existing interactive
  CPU job451363 untouched. Assigned H200s report143771MiB, zero initial compute
  processes/memory/utilization; per-job UUIDs and telemetry retained.
- GPU smoke and formal-launch outcomes are recorded below after verification.

- First five GPU smokes all completed A where applicable plus50B/50C updates,
  then rejected bitwise next-update replay. Dedicated diagnostic451672 found
  exact EMA/scheduler/CPU+CUDA RNG and only two B weight tensors differing,
  maximum1.862645149e-9; this was a post-update numerical difference rather
  than evidence of lost checkpoint state. Original Full does not enable
  deterministic CUDA algorithms; its settings remain unchanged.
- Strengthened smoke to separately require exact checkpoint restoration of
  raw/EMA/Adam/scheduler/RNG before replay, followed by native GPU next-step
  rtol1e-6/atol1e-8 with recorded model/EMA/optimizer differences. CPU exact
  continuation remains covered. Retry five independent jobs451728--451732,
  frozen `source_final`, `smoke_final/{B,C,D,E,F}` receipts. Failed receipts
  and source snapshots retained; none of these smoke weights seed formal runs.
- Compared current Full with archived November3 L1 forward using identical
  weights and denoising samples: all A/B/C outputs, losses and parameter
  gradients exactly equal. Receipt `full_legacy_parity.json` in development
  evidence directory. No Teacher, K64, Full weights or training settings changed.

- All five final H200 smoke gates PASS: each50B/50C optimizer updates, plus2A
  for C/D/F; exact checkpoint state/RNG recovery; native next-step FP32 close;
  portable EMA prediction shape[1,8,7]. Maximum next-step model differences
  B3.7253e-9,C7.4506e-9,D/E/F1.49012e-8; all EMA differences zero. All eight
  workers pinned to allocated cores; batch32 and full-width models retained.
  C smoke throughput includes a full5GB checkpoint write plus state hashing,
  so it is not a steady-state production throughput estimate.
- Additional initial-action parity test passes: Full and B--F start with
  exactly identical conditioning and per-sample denoising loss. This complements
  the surviving-weight checks despite subsequent different computation graphs.
- Formal independent jobs submitted after all five PASS receipts:
  B451818,C451819,D451820,E451821,F451822. All `yangyc/user_wangzihao`, oneH200,
  32CPUs/384GiB, gpu,gpux common48h (queried gpu100h/gpux48h), Requeue=1,
  Dependency=null. No overload long jobs. Source and all job definitions
  captured in registry.json/submitted_jobs_verified.json. Formal outputs are
  root/{B,C,D,E,F}; source_final is immutable. Formal models start fresh.

| Variant | Total parameters | Stage C trainable |
|---|---:|---:|
| B | 314058495 | 314058375 |
| C | 313806664 | 313806544 |
| D | 314168200 | 314168080 |
| E | 313670207 | 313670087 |
| F | 314236552 | 314236432 |

Stage-specific active-gradient counts and head breakdowns are in each
SMOKE_PASS.json and formal gradients.jsonl;120 frozen normalizer parameters
explain the total/C-trainable difference. B/E E1 explicitly records skipped A.

- 23:01CST startup verified: all five formal jobs RUNNING concurrently, B/C/D
  server13,Eserver30,Fserver29. Every run completed its first real optimizer
  update and atomically saved latest plus an external first_update archive.
  B/E E1 zero-update checkpoints retained. All five are loading/hash-checking
  the full299494742868-byte RAM cache before subsequent updates; RAM-ready
  completion is not yet claimed. F formal permutation exactly matches smoke.
  `startup_verified.json` records states, nodes, accounts, checkpoint sizes and
  archive paths; all BatchFlag=1,Dependency=null,Requeue=1. Jobs are independent
  of the user's computer and the interactive CPU session, which was untouched.

- 2026-11-12: user requested C continuation after job451819 was CANCELLED by UID0 at02:16:47. Accounting confirms prior successful requeue on November11; canceled attempt stopped during full RAM-cache loading, without SAFE_STOP or a Python/OOM error. MaxRSS304627592KiB vs384GiB requested; cancellation cause remains unknown. Submitted replacement477565 using original registry command, frozen source_final, original output/latest.ckpt and all original settings; verified gpu100h/gpux48h common48h, yangyc/user_wangzihao, Requeue=1,32CPUs/384GiB. Job RUNNING on server13/H200; restored epoch484 and completed/saved step986616 (previous986615). Full RAM readiness not yet verified. Other jobs untouched. Receipts: experiment-root/resume_C_477565.json and resume_C_477565_verified.txt.
- 2026-11-12 10:35CST loss snapshot: generated completed-epoch means (deduplicated by epoch, no smoothing) for BCDEF, separating action and utility objectives. Last completed epochs B557/C483/D551/E571/F572. Common epochs464--483 mean action loss B0.00320031/C0.00321059/D0.00326753/E0.00325133/F0.00325086. Artifacts/script/CSV/summary: `/data/wangzihao/outputs/manifeel/bcdef_loss_review_20261112/`. These are training losses only; F utility uses shuffled targets, B/E have no utility objective. All five jobs running at snapshot; C477565 has begun per-array RAM loading, full readiness still unverified.
- 2026-11-12: prepared user-requested ModelScope release under `ggkiller/UVIP/ManiFeel/ablation/`: 34 fixed epoch20/50/100/200/300/400/500 checkpoints plus README,170844094786bytes total; only C epoch500 is not yet generated. Descriptive directories: no_utility/no_film/no_utility_query/no_utility_no_film/shuffled_utility. Hardlink staging, manifest and direct HTTPS uploader at `/data/wangzihao/outputs/modelscope_uploads/UVIP_manifeel_ablation_20261112/`. Remote repository is accessible and destination does not yet exist. No upload or content hash computation performed: user explicitly disallowed hashing; asked whether protocol-required SDK hashing is acceptable while omitting independent verification. Awaiting answer before transfer.
- User subsequently allowed protocol-required hashing. Persistent direct API upload started; recovered expired local ModelScope cookie using the existing saved credential, then restarted authenticated uploader PID961544. Approximately44GB sent at last foreground check, no new errors. User requested silent background operation; uploader continues detached with retry and final filename/size audit, writing `upload.log` and `SUCCESS.json` in the release work directory. C epoch500 still absent and not included; no claim of upload completion yet.

- ModelScope ablation upload completed2026-11-12 11:16:13CST:34checkpoints+README,170844094786bytes,zero failures/deletions. SUCCESS.json recorded filename/size matches; on user status request independently re-listed remote and reverified all35planned paths/sizes. C epoch500 remains ungenerated and unpublished.
- 2026-11-12 ~13:30CST: user reported zero_gpu_util/low_sm_util for C477565. Job still RUNNING; no STOP_REQUEST/SAFE_STOP, no stderr error, latest step986616. GPU telemetry0%/42063MiB. RAM load advanced45->46arrays,207.28->213.79GB of299.49GB during diagnosis, so not fully stalled. Trainer926653 has8busy loader threads (~94%CPU each),~253GB RSS,zero process swap; cgroup usage~260GB/412GB,failcnt0. Process accumulated237.79s user vs80529.7s system CPU;10s sample added0s user/70s system CPU. Node memory PSI full~25%,high historical direct-reclaim/compaction counts,THP madvise; points to kernel/memory-loading bottleneck rather than GPU computation or job-limit OOM. Exact kernel hotspot not verified (thread stacks permission denied); no live process/global VM settings changed. Prior root cancellation cause remains unproven.

- 2026-11-12 user authorized immediate training during incremental cache loading. Implemented opt-in streaming cache: all decoded arrays mapped read-only, original sampler/order retained, bounded8MiB sequential background hash/read warms shared OS page cache; no299GB anonymous copy or all-cache barrier. Per-window validation remains active; background failure aborts before subsequent updates. NUMPY_MADVISE_HUGEPAGE=0 for streaming jobs, no global VM changes. Explicit source-only transport migration verifies every other original training-contract field and records actual runtime provenance separately. Immutable deployment snapshot source_streaming_20261112; original source_final untouched.
- Validation:18targeted tests passed; full424passed,5skipped. Real H200 preflight in477565.1 recovered986616, performed40updates to986656, saved and exited0. Mapping setup0.902s;20-update cold sample24.17samples/s.8workers individually bound within allocated32CPUs. PASS.json/affinity_gpu.json/pytest_full.log and original production checkpoint hardlink retained under streaming_preflight_C. Preflight weights excluded from production continuation. After PASS, cancelled only477565 to replace its stalled eager loader; production continuation submission recorded separately.

- Replacement production C480372 RUNNING on server26/H200, original yangyc/user_wangzihao,32CPUs/384GiB,48h,Requeue=1. Resumed original986616; mapping setup0.17655s; latest verified step986640, recent30.23samples/s, GPU sampled69% at13:45:43CST while background cache scan still progressing. No stderr errors. Receipt resume_C_streaming_480372_verified.json. Exact performance remains cache/node dependent; other production jobs untouched.
