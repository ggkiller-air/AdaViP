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
