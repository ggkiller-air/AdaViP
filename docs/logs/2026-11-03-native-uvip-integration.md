# Native advanced-model UViP integration

User final scope: FTP-1 six independent single-task Students and Tabero-VTLA
one multitask Field+FS Student, reusing fixed released Teachers. All inherited
vision/VLM/tactile parameters stay frozen, including Tabero VLM LoRA. Only
Action Expert LoRA plus new context/utility/FiLM modules learn. No additional
attention, query, action residual, or controller changes. Utility/context and
action/utility gradients are separated. C keeps B's freeze range and lowers LR.

- Added native FTP subclass and rank16 LoRA on126 original action projections.
  Strict release coverage accounts for the tied VLM embedding and explicitly
  unused native expert lm_heads. Six actual checkpoint checks pass:1142 tensors,
  exact full-forward loss/native cached sampling at initialization, all9
  Full/Masked interventions, state-adaRMS bypass exclusion, gradient whitelist,
  exact parameter/Adam/RNG continuation. Trainable total10,898,824.
- Native-parser data audit passes exact decoded pixels and state/action/mask
  values for six tasks,50episodes each,67040frames total, matching release stats.
  Preserved the pinned parser's executable BGR224 convention, current-frame
  target timeline, mix arm deltas and absolute gripper slot44. DP next-frame
  caches/protocols are separate and unchanged. Native arm subtraction promotes
  to float64 before normalization; mask observations require floating dtype.
- Added native JAX Tabero subclass,6utility/3FiLM, original H50x32 force-aware
  weighted_full objective (7position,6force@0.1,19padding@1), Field TCН/history.
  Shape-strict release restore, identity full loss/sampling, all7 interventions,
  state removal, gradient isolation and10action-LoRA tensor whitelist pass.
  Existing action LoRA uses FP32 masters; VLM LoRA/TCN/projections remain frozen.
  Trainable total26,545,926. Actual trainer A2/B2/C2 with real labels passes exact
  interrupted-at3 vs uninterrupted6 selected-parameter comparison. Teacher
  adapters restored before continuing labels; diagnostic checkpoints excluded
  from formal Stage2 status.
- BF16 masked KV vs full forward differs numerically. Both formal scorers use
  full native attention and reuse only frozen prefusion encoder tokens. Labels
  preserve signed native loss differences, shared per-window noise and K8
  stratified native Beta times. RMS scales fitted only after all labels finish.
- One job per Student automatically labels then runs A/B/C: FTP1000/9000/10000,
  effective64/micro8,rank16,LoRA LR5e-5; Tabero2000/28000/20000,effective32/micro4,
  existingrank32,LoRA LR2.5e-5. New UViP/utility LR1e-4, B/C utility coefficient
  0.05, C LR multiplier0.1. Latest every100, all periodic1000/2500 retained.
  FTP EMA0.99; native Tabero no EMA. Checkpoints include adapters, optimizer,
  config/Teacher/label identity and RNG or deterministic step PRNG. External
  first-Student archive configured. gpu,gpux48h/requeue/flock/soft stop and
  continuation preserve source snapshots and committed label shards.
- yangyc parent56GPU cap caused AssocGrpGRES despite idle H200. Verified user's
  existing overload association and QOS allow these partitions; new jobs use
  that association, untouched existing DP staysyangyc. FTP put_bottle yielded
  after248labels for Tabero, automatically requeued422921 without lost shards.
- Active jobs: FTP422867/422882/422883/422884/422885, queued422921; Tabero422917.
  Source roots: FTP `/data/wangzihao/experiments/advanced_uvip/20261103_1733`;
  Tabero `/data/wangzihao/experiments/advanced_uvip/20261103_1747`.
  Canonical registry `/data/wangzihao/experiments/advanced_uvip/20261103_1707/registry.json`.
  GPU server13, UUID479e80e2... / da8386db... for initial pilot checks; each
  job records its assigned UUID, empty initial memory, CPU affinity and telemetry.
- Root progress document: `/public/home/wangzihao/projects/UVIP_5_Routes_Progress_2026-11-03.md`.
  Critical-event callbacks plus CPU watcher refresh it; formal Stage2 remains
  pending until labels complete. No formal Student rollout performed. Original
  three DP runs/weights/labels and pre-existing uncommitted ManiFeel files untouched.

Additional checks:4FTP module pytest tests pass in new environment, Tabero DP
tests skip there for absent diffusion_policy; the full existing366passed suite
and dedicated10tests previously passed in the declared ManiFeel environment.
Both native integrations now have stronger real H200 checkpoint checks above.
Python syntax checks and sbatch bash syntax checks pass.

- User requested first-table ready steps display `human` instead of `pending`.
  Updated Markdown and progress refresher: an unstarted stage immediately after
  a completed stage becomes human; running/completed states stay unchanged.
  CPU watcher reloaded to preserve this rule; native GPU jobs untouched.

- User reports administrator explicitly permits loading a buffer to satisfy
  minimum15%VRAM. Measured H200143771MiB: DP24319MiB already16.9%, advanced
  scorers15949–17043MiB below threshold. Reused existing bounded, zero-filled
  holder in separate overlapping srun steps belonging to each GPU allocation;
  verified reservation records contain the corresponding GPU job IDs, not CPU
  supervisor ID. Hold16%standalone (22.368GiB) so stage transitions stay covered.
  Verified totals: FTP39601MiB,put_bottle39471MiB,Tabero40565MiB (27.5–28.2%).
  DP unchanged. CPU supervisor423254 follows registry IDs every30seconds,
  including queued insert_hole423196 when allocated; holders release on formal
  completion or per-job soft-stop and are isolated from model/optimizer/RNG/
  checkpoint. Measurements at registry root gpu_memory_guard/verification.json.

- 19:46: user requests starting Stage2 from published ManiFeel utility labels.
  Verified final ModelScope targeted API has255shards and7aggregates, all
  SHA/size equal local complete65265-window E50L1 K64 output. FixedTeacher
  SHA95bd523e...fa5fb, EMAE50/55700; exact48-tensor archivednormalizer.
  Implemented separate Stage2L1Policy preserving sixsensor/late-language
  encoder, newlanguage-actiontokenprojection, inheritedS0head/freshDPactor;
  generic diagnostics and evaluation restore handle L1 without affecting L0.
  RealCPUpreflight label/cache/source/Teacher/encodercontext exact passed.
  27 targeted tests passed inclA/B/C gradients/exactrawEMAAdamRNGresume/
  portableprediction. Fullsuite running separately. Submitted423348 one
  H200/32CPU/384G/gpu,gpux/48h/requeue, frozen source under experiments/
  manifeel/l1_e50_uvip_s0_b32_ep600_seed42_20261103_194431. PendingPriority,
  GPUcorrectness/performance awaitallocation. Has isolated16%VRAMbuffer,
  keepsallselectedperiodic+E600/latest and archivesfirstformalupdate.
  registry.dp_stage2_runs tracks separately from7advanced runs.
- Full regression completed:368passed,2skipped (180.33s). CPUprogress
  watcher423354 reloads DPStage2 tracking; GPUjobs untouched.
