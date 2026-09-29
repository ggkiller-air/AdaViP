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

## 2026-09-27 Continuations

- USB insertion `321171` and gear assembly `321172` completed. Object search
  `321176` and ball sorting `321177` were still running at the continuation
  check.
- Jobs `321170`, `321173`, and `321174` were cancelled by the cluster's
  360-minute GPU low-memory rule (6% occupancy, below its 10% threshold),
  according to cluster notifications. Peg reorientation `321175` was also
  cancelled after about six hours, but no matching kill notification was
  available. These were not Slurm four-day wall-time expirations.
- Submitted the same batch-8, single-task launcher with resume enabled:
  `323095` peg insertion from `latest_epoch200.ckpt`, `323096` peg
  reorientation from `latest_epoch200.ckpt`, `323097` nut-bolt assembly from
  `latest_epoch100.ckpt`, and `323098` bulb installation from
  `latest_epoch100.ckpt`. Each job selects its own GPU through Slurm; no node
  placement options were used. Further continuation may be needed if the
  low-memory rule terminates a run before epoch 301.
- Copied these four latest periodic checkpoints outside their active run
  directories to `/data/wangzihao/checkpoints/manifeel/continuations_20260927/`.
- Object search `321176` was subsequently killed at 21:36 CST by the same
  low-memory rule (6% occupancy for 364 minutes), during epoch 228. Submitted
  continuation `323588` with the original single-task batch-8 launcher and
  automatic resume from `latest_epoch200.ckpt` (next epoch 201, target 301).
  Archived that checkpoint as `object_search_latest_epoch200.ckpt` in the
  same external continuation archive.

## 2026-09-29 Remaining Two Tasks

- Verified six completed tasks through Slurm completion and final
  `latest_epoch301.ckpt`: peg insertion, USB insertion, gear assembly, peg
  reorientation, object search (`323588`), and ball sorting.
- Nut-bolt continuation `323097` and bulb continuation `323098` were killed
  on 2026-09-28 at 03:31 and 03:06 CST by the 360-minute GPU low-memory rule.
  Their logs stopped during epochs 224 and 235 respectively; both retained
  `latest_epoch200.ckpt` as their newest periodic checkpoint.
- After verifying no active duplicates, submitted `326768` for
  `nut_bolt_assembly` and `326769` for `bulb_installation` using the original
  single-task launcher. Both keep batch 8, 301 target epochs, automatic resume
  (next epoch 201), one GPU, 64 CPUs, 512 GiB RAM, and four days. Both were
  pending for `Priority` after submission, with no node-placement options.
- Archived each epoch-200 recovery checkpoint, Hydra configuration directory,
  and current training log under
  `/data/wangzihao/checkpoints/manifeel/continuations_20260929/<task>/`.

## 2026-09-29 Curves and ModelScope Archive

- Verified final epoch-301 checkpoints for all eight single-task runs. The
  Power-Plug fusion-hypernet run (`dp_power_plug_hyper_resnet_fusion_hypernet_batch8_ep400_seed42`,
  Slurm name `mf-plug-hyper-fh-b8`) has checkpoints through epoch 300 and its
  log currently reaches epoch 341.
- Generated per-run and combined train/validation loss plots through epoch 300
  with logarithmic y-axes under
  `/data/wangzihao/outputs/manifeel/0929ours/loss_curves_log/`.
- Validated all 27 epoch-100/200/300 checkpoints as readable workspace archives,
  then uploaded them directly with all proxy variables unset to
  `ggkiller/multi-fm` revision `master` under
  `0929ours/<task>/latest_epoch{100,200,300}.ckpt`. A direct remote listing
  confirmed 27 files and matching byte sizes.
