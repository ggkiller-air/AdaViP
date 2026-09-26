# 2026-09-26 Power Plug ResNet Fusion Only

- Added a Power Plug TacRGB `ResNetFusionObsEncoder` with independent trainable
  wrist and right-tactile ResNet18 encoders, baseline GroupNorm and ImageNet
  input normalization, and vision-led attention fusion. It has no HyperNet or
  FiLM modules, preserves the baseline feature order and 1,031D condition,
  and starts with a zero fusion gate so its initial output matches the
  upstream baseline.
- Reused the existing Power Plug CS fusion-only job `320004`: frozen CLIP RN50
  and Sparsh DINO ViT-B with trainable projection/attention, without FiLM or
  HyperNet. It remained pending for `AssocGrpGRES`; no duplicate job was
  submitted.
- The new ResNet fusion run uses the upstream Power Plug TacRGB training
  protocol with 50 demonstrations, seed 42, train/validation batch size 8,
  400 epochs, validation every 10 epochs, and a checkpoint every 50 epochs.
  The retained workspace keeps every periodic checkpoint and saves epoch 400.
- Focused regression tests passed (9 passed), including baseline-equivalent
  initial output and gradient flow into both ResNets and attention. The full
  repository suite passed (61 passed, 1 skipped); actual Hydra instantiation
  produced two trainable ResNet18 branches, a 1,031D output, and no HyperNet.
  Shell syntax, Python compilation, and `git diff --check` passed.
- Allocation `319633` on `server34` was CPU-only, so local GPU memory,
  utilization, and process details were unavailable; the `gpu` partition
  advertises NVIDIA H200. Under the user's prior direct-submission
  authorization, submitted resource-only job `320014` with one GPU, 64 CPUs,
  512 GiB RAM, and four days. It was pending for `Priority` immediately after
  submission. No node-selection options were used and no training result
  exists yet.
