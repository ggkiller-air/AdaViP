# 2026-09-26 CS Fusion Power Plug and Ball Sorting

- Added the CS ablation: frozen OpenAI CLIP RN50 and Sparsh DINO ViT-B, with
  trainable projection, vision-led attention fusion, and the original diffusion
  policy. It removes the FiLM and HyperNet paths while retaining the same
  preprocessing and five-frame tactile pairing as the prior FiLM run.
- Configured separate Power Plug (`wrist`, 6D action) and Ball Sorting (`front`,
  7D action) runs on `plug_quan_Aug02` and `sorting_quan_Aug8`. Each uses 50
  demonstrations, train/validation batch size 8, seed 42, at most 400 epochs,
  validation every 10 epochs, and checkpoints every 50 epochs. The retained
  workspace keeps all scheduled checkpoints and saves the final epoch 400.
- Both real Zarr datasets produced the expected observation and action shapes.
  The ManiFeel Hydra entry point composed both configs. Focused tests passed
  (12 passed); the repository suite passed (55 passed, 1 skipped). Shell syntax,
  Python compilation, launcher dry runs, and `git diff --check` passed.
- Current allocation `319633` is CPU-only on `server34`; no local GPU or GPU
  processes were visible for an interactive preflight. The `gpu` partition
  advertises NVIDIA H200 GPUs. After the user explicitly authorized direct
  `sbatch` queueing, submitted resource-only training jobs `320004` (Power
  Plug) and `320005` (Ball Sorting), each requesting one GPU, 64 CPUs, 512 GiB
  RAM, and four days. Both were pending for `Priority` immediately after
  submission. No node-selection options were used.
