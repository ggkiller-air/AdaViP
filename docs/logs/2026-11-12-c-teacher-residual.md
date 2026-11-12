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
