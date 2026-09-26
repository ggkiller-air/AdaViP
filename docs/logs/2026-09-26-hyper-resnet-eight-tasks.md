# 2026-09-26 HyperResNet Fusion Across Eight Tasks

- Added a shared launcher for the eight non-Power-Plug ManiFeel tasks:
  `peg_insertion`, `usb_insertion`, `gear_assembly`, `nut_bolt_assembly`,
  `bulb_installation`, `peg_reorientation`, `object_search`, and
  `ball_sorting`. Each uses the verified trainable ResNet + vision/tactile
  HyperNet + attention fusion configuration, with `use_fusion_hypernet=false`.
- All profiles use the `vistac_wrist` observation schema (`wrist`, right
  TacRGB, and state), 50 demonstrations, batch 8 for train and validation,
  seed 42, 301 maximum epochs, validation every 10 epochs, and checkpoints
  every 100 epochs. The retained workspace keeps every checkpoint and saves
  the completed epoch 301 checkpoint. Six-dimensional and seven-dimensional
  action shapes are overridden per task.
- Zarr dataset directories and Isaac Gym configs were checked for all eight
  tasks. Focused regression tests passed (`9 passed`), including all task
  dry-runs; Hydra instantiated representative 6D and 7D profiles with output
  width 1,031 and all encoder parameters trainable. Python compilation, shell
  syntax, and `git diff --check` passed.
- Submitted eight resource-only jobs, each requesting one scheduler-selected
  GPU, 64 CPUs, 512 GiB RAM, and four days, with no node-selection options:
  `321170` peg insertion, `321171` USB insertion, `321172` gear assembly,
  `321173` nut-bolt assembly, `321174` bulb installation, `321175` peg
  reorientation, `321176` object search, and `321177` ball sorting. All were
  pending for `AssocGrpGRES` immediately after submission.
