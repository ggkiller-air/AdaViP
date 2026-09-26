# 2026-09-26 Power Plug Fusion HyperNet

- Added an optional `fusion_hypernet` to `HyperResNetObsEncoder`. It generates
  gamma/beta for the vision-led fusion feature after the existing vision and
  TacRGB HyperNet paths, with a separate positive residual scale initialized
  at 0.01 and zero-initialized output so the initial encoder remains baseline
  equivalent.
- Added the matched Power Plug `fusion_hypernet` profile: 50 demonstrations,
  batch 8 for train and validation, seed 42, 400 maximum epochs, validation
  every 10 epochs, and checkpoints every 100 epochs. The retained workspace
  disables pruning and saves the completed epoch 400 checkpoint.
- Focused tests passed (8 passed); the full repository suite passed (63 passed,
  1 skipped). Hydra instantiated the new encoder with output width 1,031 and
  all ResNet, attention, and HyperNet parameters trainable. Shell syntax,
  Python compilation, dry-run launchers, and `git diff --check` passed.
- Current allocation `319633` on `server34` is CPU-only; `nvidia-smi`
  reports no devices or GPU processes. After the user's direct submission
  authorization, submitted resource-only job `321116` using one GPU, 64 CPUs,
  512 GiB RAM, and four days, without node-selection options. It started on
  `server18` with an NVIDIA H200; the launch log shows the expected dataset,
  CUDA device, and training loop toward epoch 400.
