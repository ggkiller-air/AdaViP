# C_teacher_residual: Teacher policy retention experiment

- Base: pushed367317d7d9260a16ddd71e6d44838c3fe0bc2ae8. No newer remote commit at fetch.
  Isolated branch experiment/c-teacher-residual-20261112; local C streaming-cache
  updates copied unchanged into baseline commit2f1a4a9. Original worktree and C480372 untouched.
- One candidate only: canonical C initialization, then strict L1E50 Teacher EMA
  encoder/normalizer/U-Net inheritance. Conditioning is z + w_out(h), 256->256,
  zero weight/bias, no normalization after addition. Existing C normalization,
  query, FF/attention, utility/K64 and both detaches retained. RNG fork prevents
  the new linear initializer from shifting common weights or training draws.
- Canonical C hash21db0ba749cc21041909ec85f68de14bae63f7693063b72eb03929a22b4fbf03
  required before copying Teacher U-Net. w_out joins existing action AdamW group,
  frozen throughout A; EMA, strict state, and full resume include it.
- Entry: scripts.manifeel.train_stage2_l1_s0 --ablation C_teacher_residual;
  allocation wrapper slurm/manifeel/train_stage2_teacher_residual.sbatch.
  Batch32/seed42/2040updates per epoch/A1/B2--6/C7--600, global clip1, all original
  rates/EMA/sampling/data identities retained. Current C streaming data transport.
  Fixed snapshots E1/6/10/20/50/100/200/300/400/500/600; E200/E400 designated comparisons.
- Smoke requires nine-task context, residual, denoiser and full100-step sampling
  bitwise Teacher parity; A/B/C parameter updates and both detach routes; zero
  upstream residual gradients initially, nonzero after temporary w_out update;
  exact raw/EMA/Adam/scheduler/RNG restore and deterministic next update;
  portable raw/EMA full sampling. Deterministic kernels are smoke-only; formal
  numerical settings remain those of C. No rollout on H200.
- Verification and submission receipts follow below.
- CPU regression426passed/5skipped; first focused suite15passed. First H200
  smoke484586 rejected full-sampling equality after context and denoiser passed.
  Located test asymmetry in installed diffusers: add_noise moves alphas_cumprod
  to the action device in-place. The helper called it only on Teacher, comparing
  CUDA scalar arithmetic against Student's fresh CPU scheduler. Validation now
  checks full sampling before either add_noise, then after identical add_noise
  on both; restores transient schedulers afterward. No policy/training/sampling
  implementation or tolerance changed. Raw portable inference likewise checks
  both fresh/fresh and warmed/warmed scheduler states. Failed receipt retained.
- Final H200 smoke484651 PASS (server13/H200 GPU-c27a2db2-cfc6-ccac-4c30-cc445854d4f8):
  nine tasks x100 sampling steps exact, both fresh and warmed schedulers. A2/B50/C50
  real batch32 updates; frozen roles unchanged, all intended active roles changed.
  Raw/EMA/Adam/scheduler and RNG restore plus next update bitwise exact (maxdiff0).
  Portable raw and EMA predictions exact. Eight workers pinned inside32allocated
  CPUs. B234.56samples/s; C70.12includes checkpoint write/state hashing, not a
  steady-state estimate. Scheduler-fix CPU tests2passed; earlier full426passed/5skipped.
- Immutable training source17a2b98 (source_v2), Python source fingerprint
  2b6e0e4fbe1d530a138dfe2ee3e2b4786bd2622dd6c8a7b54ea193db2870848b.
  Root `/data/wangzihao/experiments/manifeel/c_teacher_residual_b32_ep600_seed42_20261112/`.
  Full commands, dependency commits, resource snapshots and all receipts in registry.json,
  source_provenance.json, validation_summary.json and smoke_v2/SMOKE_PASS.json.
- Formal job484745 RUNNING server13/H200, yangyc/user_wangzihao, gpu,gpux,
  32CPUs/384GiB, common48h (gpu100h/gpux48h queried), Requeue=1, no dependencies.
  Output root/C_teacher_residual. Starts fresh, excludes every smoke weight.
  Verified step80/E1/A,80actual updates; cold-data throughput20.98samples/s.
  Streaming mapping0.19964s,299494742868bytes background-verified; full scan
  completion is not yet claimed. No stderr errors at startup audit.
- First recoverable checkpoint step1 is2512016112bytes, archived outside run at
  `/data/wangzihao/archives/manifeel_uvip/c_teacher_residual_b32_ep600_seed42_20261112/C_teacher_residual/first_update.ckpt`.
  Independently loaded it: both raw/EMA U-Nets have Teacher SHA5252d628...778e9f,
  both w_out weights/biases remain zero, full resume keys present.
- Pure-CPU publisher484746 runs independently of training (server40,1CPU/4GiB,
  cpu5day limit with checkpoint-independent requeue). Delivery source52da934.
  Copies every fixed epoch checkpoint to an external archive before uploading to
  `ggkiller/UVIP/ManiFeel/ablation/C_teacher_residual/`; filenames/sizes verified
  via API, SDK transport hashing retained. README/source fingerprints/smoke evidence
  already published. first_update.ckpt is a wiring check at protocol step1;
  it is not an E200/E400 result. E200/E400 remain ungenerated at this record.
- No simulator rollout, changes to existing jobs, B--F retraining, extra shuffle,
  utility-free control, parameter search or score-driven schedule changes.
- Publication confirmed: first_update.ckpt2512016112bytes plus its verification
  receipt now present remotely. It remains an E1/step1 initialization-interface
  artifact, not a completed protocol epoch. Strict contract comparison against
  current C passed: every shared field matches except variant/initialization
  audit/source provenance/fingerprint; receipt root/config_equivalence.json.
- User requested two-H200 acceleration, then also four-H200 queued timing.
  Prepared isolated same-node DataParallel compute screens, global32 retained
  (16/card or8/card), global noise/timesteps generated once on primary before
  scatter. Global per-example/per-dimension loss gathering avoids unequal-tail
  mean weighting. Bare model/optimizer/EMA remain primary-owned. CPU adapter
  tests9passed for A/B/C and batch1/2/5, exact loss reduction and RNG accounting.
  Existing formal484745 remains untouched. Both requests use one frozen source
  and checkpoint, ABBA single/parallel timing, same Adam/EMA/global clip1.
  A >=1.15 compute speedup only nominates further full-epoch/resume validation;
  it does not automatically migrate training or claim end-to-end acceleration.
- User scheduling preference: all debugging/smoke/throughput/cache audits use
  overload/overload; only required-result long training avoids overload. Recorded
  in /public/home/wangzihao/AGENTS.md. Formal484745 stays yangyc/user_wangzihao.
- Initial normal-account screens485455(two)/485456(four) both stopped before
  timing on single-batch32 vs split-batch gradient relative-L2 gate1e-4:
  2.3274e-4 and4.7724e-4 respectively; max absolute3.9954e-6/7.4004e-6.
  Production untouched. Add a matched-shard serial single-GPU diagnostic to
  distinguish batch-shape arithmetic from replica/scatter/reduction errors;
  retain1e-4 gate for that comparison, record original differences explicitly.
- Data review: production scanner completed all63arrays/299494742868bytes in
  1014.35seconds. Code joins every complete label to task/episode/timestep/source
  index, preserves masks and Teacher normalization; mapped arrays are read-only.
  Found missing upfront geometry check: unlike verified eager path, streaming
  fallback could resize a differently sized cache again. Harden validator to
  reject image geometry, record identity/contract, total-byte and task-count
  mismatches. Live frozen source is unchanged; audit verifies whether current
  artifacts trigger any of these guards. New bounded CPU audit compares original
  Zarr and mmap at every episode's first/last complete window plus task-stratified
  random windows, normalized values, global sampling/resume suffixes, worker PSS,
  and independently hashes the complete cache. Jobs use overload.
- Overload CPU485669:15tests passed;853raw-vs-mapped windows across400episodes
  bitwise exact for all fields and normalized observations/actions. All65265label
  indices and epoch1/2/7/200/400/600 sampler/resume suffixes pass; tail17 retained.
  Worker PSS_Anon~113MiB each, mapped-file pages~3.5GiB each in this subset audit,
  no299GB anonymous per-worker copies. Current cache geometry passes hardened checks.
- DataParallel485667/485668:matched-shard gradient relative L2~7.97e-9/3.37e-8,
  isolating earlier difference to full-batch vs shard-size arithmetic. Compute
  screen rates single~201/204,2GPU~89,4GPU~67samples/s; no production promotion.
- CPU485669 later received SIGTERM after31arrays/135151175040bytes; no hash mismatch
  or OOM message. Original production full63-array hash pass remains valid.
  Retain completed prefix and finish remaining arrays in bounded overload shards.
- Added persistent DDP screens485838/485839. Both matched-shard gradients passed,
  but rank-state equality rejected after first timed block. Located benchmark-only
  reset alias: AdamW.load_state_dict can retain CPU step tensor references, so the
  rank0-only single baseline mutated its reusable CPU snapshot. Deep-copy optimizer
  snapshots on each benchmark reset; regression verifies saved steps/moments remain
  unchanged. Formal trainer reloads fresh checkpoints and is unaffected. Failed
  receipts retained; no throughput acceptance or production migration from them.
- Corrected overload DDP screens485957(two)/485958(four) both completed, with
  updated replicas bitwise equal. Matched-shard gradient relative L2 is6.74e-9
  and4.46e-8. Global batch32, frozen step10200 checkpoint, ABBA timing50updates
  after10warmups per block, including global clip/AdamW/EMA, excluding data load
  and checkpoint I/O. Two: baseline200.23 vs DDP200.59samples/s (1.00179x).
  Four: baseline203.85 vs DDP204.77samples/s (1.00446x). Neither passes1.15x
  nomination threshold; keep formal single-GPU and do not expand experiments.
  Reports: root/parallel_bench_20261112/ddp{2,4}_v2/REPORT.json.
- Optimizer snapshot regression plus adapter checks:10passed in28.99seconds
  on overload CPU485955. Production source remains17a2b98; review/benchmark
  fixes are committed separately at8df8192 and do not alter the ongoing run.
- Formal snapshot at2026-11-12 22:36CST: E13,24680optimizer updates, stageC,
  finite task losses; completed E7--12 throughput197--204samples/s. At this
  rate E200 is roughly17--19hours away, E40035--38hours, E60053--57hours,
  excluding requeue delay and3090 rollouts. Existing signal/safe-stop/full-resume
  requeue path handles the48h allocation boundary. No early-score schedule change.
- Independent full-cache audit completed: overload CPU485955/485956 exited0
  in8m41s/4m04s. First31arrays from interrupted scan plus32resumable hash
  receipts cover all63arrays/299494742868bytes. Finalizer logged
  FULL_CACHE_AUDIT_PASS; independently rechecked every receipt SHA against
  manifest and all current dev/inode/size/mtime identities, all pass.
  Output root/parallel_bench_20261112/cache_audit/PASS.json with853window
  parity, normalization/sampling/memory evidence. No current cache corruption,
  index drift or duplicate normalization was detected. Raw-vs-cache comparisons
  sample853windows spanning all400episodes; hash checks cover all cached bytes,
  and index joins cover all65265label rows. Geometry guard regression passed;
  guard addresses a latent invalid-cache acceptance path, not a demonstrated
  problem in current training data. No production restart is needed.
- Follow-up to scaling question: overload two-H200 profile486147 completed0
  in1m45s onserver13 GPUs5/6, initially idle, NVLink-connected. Source53cd2ae,
  root/parallel_bench_20261112/source_profile; reports ddp2_profile/profile_rank*.json.
  Adds CUDA event phase boundaries without per-phase synchronization; host launch
  intervals recorded separately. CUDA intervals include idle/launch gaps and are
  not pure kernel time or an isolated NCCL measurement. Inputs already reside
  on GPU; data loading/checkpoint I/O excluded. Same global32, checkpoint, RNG,
  loss and optimizer; matched-shard gradient6.69e-9, updated replicas exact.
- Across ABBA blocks rank0 mean CUDA phase ms single vs2DDP: forward/loss40.12
  vs37.88; backward+global clip84.07 vs86.26; AdamW/scheduler+EMA33.57 vs35.12.
  Wall step159.49 vs159.97ms, compute speedup0.99697x (no material change).
  Model313872456parameters,313872336trainable, FP32 trainable bytes1255489344.
  This establishes poor batch-size scaling of forward and unchanged full-model
  update costs, with backward/synchronization failing to provide savings. It
  does not isolate allreduce duration or prove CPU launch overhead dominates;
  those require a kernel/communication trace. RAM loading cannot explain this
  GPU-resident compute screen. Keep formal source/protocol/job untouched.
- User expanded scope to larger batches, explicitly selected1,224,000actual
  optimizer updates with comparisons at408,000/816,000, then requested multiple
  groups. Retain original B32 job; new comparisons are B64/two GPUs, B128/four,
  B256/four. No LR scaling, architecture change or shortened cosine schedule.
- Overload batch screen486234 plus singleton retry486292: seven real-data compute
  cells passed finite losses, matched-shard gradients and replica equality. B32/1
  =200.37samples/s,159.70ms/update; B64/1=298.40,214.48ms; B128/1=385.58,331.97ms;
  B64/2=381.45,167.78ms; B128/2=583.62,219.32ms; B128/4=760.51,168.31ms;
  B256/4=1125.59,227.44ms. Measured allocated GPU peaks for selected groups are
  15.91/16.41/26.73GiB. Sourcea08ec7c plus singleton file-store fix32eddc8;
  first singleton failed TCPStore EADDRINUSE before model creation, receipt kept.
  Reports root/parallel_bench_20261112/batch_sweep/SUMMARY.json. Pure compute,
  not convergence evidence or end-to-end timing. Equal updates imply2/4/8times
  sampled examples and increased GPU-hours; projected compute-only57/57/77hours.
- New experiment root `/data/wangzihao/experiments/manifeel/c_teacher_residual_equal_updates_20261112`.
  Added fresh canonical Teacher-residual initialization with strict SHA checks;
  update-clock A ends2040, B ends12240, C ends1224000. Original2040x600
  LambdaLR,500-update warmup/encoder ramp, AdamW, global clip and EMA unchanged.
  Data epochs/cursors are independent of reference_epoch=updates/2040.
  B64/B128/B256 have1020/510/255updates per data epoch: matching originalE200
  requires400/800/1600data epochs. Fixed reference checkpoint names retain
  original publication cadence; metadata explicitly identifies both clocks.
- DDP keeps global42+data_epoch task-balanced plans, contiguous nonduplicating
  shards, globally sample-weighted unequal tails49/113/241, independent derived
  rank CUDA streams42+rank saved/restored with CPU/Python/NumPy RNG. Rebuild DDP
  at freeze transitions, preserve fixed gradient buckets, exact raw/EMA/Adam/
  scheduler/cursor/statistics restore. Source-indexed read-only mmap reattaches
  by filename under spawn workers, with identical file-identity checks across ranks.
  Disjoint allocated CPU cores for each main rank and its workers. Original B32
  frozen source and jobs remain untouched. CPU tests13passed, covering complete
  65265-row sampler plans/resume suffixes and unequal-tail gradient weighting.
- Initial smoke486490/486491/486492 superseded before meaningful training to fix
  overlapping rank-worker affinity inherited from single-GPU helper. V2 smoke
  486546(B128) completed actual A/B/C updates, one complete data epoch including
  tail113, and fresh-process continuation with bitwise raw/EMA/Adam/scheduler/
  all-rank RNG/cursor/task-statistics equality. Receipt smoke_b128_v2/SMOKE_PASS.json.
  Cold-node B64/B256 full-data smokes were stopped after finite short updates:
  random mmap reads before sequential warming were too slow for bounded debugging.
  Metadata review also added required encoder_variant=l1/variant fields for the
  real3090 evaluation loader, plus raw/EMA100-step portable inference checks.
- Final bounded smoke sourcea4de333(source_v4): A2/B2/C20 real updates, actual
  final batch and next-epoch samples, then independent-process exact next2updates;
  all65265 sampler rows remain exhaustively checked on CPU. B128's full-data
  epoch evidence is retained separately. No tolerance relaxation. Overload jobs
  486686/486687/486688 validate B64/B128/B256 with512GiB RAM. Formal launcher
  refuses missing same-batch/world smoke, resume or portable-inference passes.
  The three formal submissions and publication receipts will be recorded below.
- V4 passed parameter routing and full fresh-process resume, but correctly failed
  the portable raw inference assertion. Diagnostic486729 found equal parameters,
  strides and module modes, but the reference retained trainable parameters while
  the evaluation loader froze them. Matching `eval().requires_grad_(False)` made
  residual conditioning and all100sampling steps bitwise identical; before that,
  conditioning differed by5.96e-8 and actions by3.79e-4. Evidence:
  `smoke_b64_v4/INFERENCE_DIAGNOSTIC.json`. No tolerance was relaxed. Final runtime
  a6424aedd2f8ed2e163dd0e636c617fb3086df33 changes only this validation setup;
  frozen source_v5 and its source.sha256 are retained and code is pushed.
- Final overload smokes486749(B64/2GPU) and486750(B128/4GPU) completed0 in8m42s
  and8m52s. Each passed13CPU checks, Teacher parity, A/B/C actual updates,
  unequal tail and data-epoch transition, exact fresh-process raw/EMA/Adam/LR/
  all-rank RNG/cursor/statistics continuation, and nine-task raw/EMA portable
  full-sampling bitwise equality. Receipts: smoke_b64_v5/SMOKE_PASS.json and
  smoke_b128_v5/SMOKE_PASS.json. B128 v2 additionally checked a whole real data
  epoch; short smokes deliberately skip the middle after checking its sampler.
- Formal B64 job486761 and B128 job486763 started2026-11-12 23:49CST onserver13,
  assigned idle H200s1--2 and3--5,7 respectively; original B32 remains onGPU0.
  Normal accountyangyc/qosuser_wangzihao,32CPUs,384/512GiB RAM,48h allocation,
  signal-triggered checkpoint/requeue and explicit full-state resume. Both
  produced first-update latest.ckpt and finite utility-only updates. Pure-CPU
  overload publishers486762/486764 use distinct B64/B128 prefixes under
  `ggkiller/UVIP/ManiFeel/ablation/C_teacher_residual_equal_updates/` and external
  archives. B64 metadata uploads verified; B128 initially hit RateLimitError
  and the existing bounded600second retry loop retained the pending work.
- B128 full-pipeline warm smoke windows measured500--704samples/s (median633),
  implying about23h to408000updates before checkpoint/queue overhead; the
  compute-only19h estimate is not an end-to-end promise. Equal-update larger
  batches test sample-rich optimization, not a matched-compute speed claim.
- Final B256 overload smoke486751 completed0 in14m40s, with every same-batch
  gate passing including the241-sample tail and bitwise portable raw/EMA full
  sampling. NFS briefly delayed visibility of SMOKE_PASS.json; submission was
  held until the actual receipt could be read and checked. Formal486770 was
  submitted23:56CST with4H200/32CPU/512GiB and is pending MaxGRESPerAccount;
  publisher486771 is running onoverload CPU. Keep this required-result long run
  in the normal queue, without displacing original jobs480372 or484745.
- Final runtime source manifest SHA256:
  b380313d5037cff6fb1b0acfa1b0b3915a62cbdc58c62a972e3ec3f494ca49c1.
  Root registry.json, per-group planned_config.json/registry.json, source_v5,
  same-batch smoke receipts and publication READMEs provide the launch contract.
  Training entry: scripts/manifeel/train_stage2_equal_updates.py; batch wrapper:
  slurm/manifeel/train_stage2_equal_updates.sbatch. Outputs are
  root/c_teacher_residual_b{64,128,256}_u1224000_seed42_20261112/C_teacher_residual.
  Remote fixed checkpoints: ggkiller/UVIP,
  ManiFeel/ablation/C_teacher_residual_equal_updates/B{64,128,256}/epoch{E}.ckpt.
  E200/E400 remain future artifacts at408000/816000updates, never early scores.
  Both running jobs completed the299494742868-byte cache verification/warm scan
  and exceeded1000finite updates with first-update latest checkpoints present.
