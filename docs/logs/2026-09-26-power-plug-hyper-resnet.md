# 2026-09-26 Power Plug HyperResNet Ablation

- Added `HyperResNetObsEncoder` as a subclass of the upstream
  `MultiImageObsEncoder`. It keeps separate trainable ResNet18 encoders for
  wrist and right TacRGB, GroupNorm, ImageNet input normalization, the baseline
  feature order, and the 1,031D condition shape. Its HyperNet modulates both
  512D streams with zero-initialized outputs and a bounded residual scale
  initialized to 0.01 (maximum 0.1). Optional vision-led attention fusion
  starts with a zero gate; both variants initially reproduce the baseline
  encoder output.
- Configured matched Power Plug `fusion` and `no_fusion` variants through the
  existing single-task DP launcher. Both use 50 demonstrations, train and
  validation batch size 8, seed 42, at most 400 epochs, validation every 10
  epochs, and checkpoints every 50 epochs. The retained workspace preserves
  every periodic checkpoint and writes the completed epoch-400 checkpoint.
- CPU tests verified baseline-equivalent initial output, gradients through
  both ResNets and the HyperNet, profile settings, launcher dry runs, and
  actual Hydra instantiation of the two ResNet18 streams. The repository test
  suite passed (59 passed, 1 skipped); shell syntax, Python compilation, and
  `git diff --check` passed.
- Current allocation `319633` on `server34` is CPU-only, with no visible GPU
  or GPU processes for interactive preflight. The user authorized direct
  `sbatch` submission. Submitted resource-only jobs `320008` (no fusion) and
  `320009` (fusion), each requesting one GPU, 64 CPUs, 512 GiB RAM, and four
  days, with no node-selection options. Both were pending for `Priority`
  immediately after submission; no training result exists yet.
