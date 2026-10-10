# Repository Guidelines

## Project Structure & Module Organization

- `adavip/model/` contains adaptive perception and transforms; `adavip/policy/`
  wraps them around an action backbone.
- `tests/` contains PyTorch/pytest tests; `configs/` stores experiment settings.
- `scripts/manifeel/` provides local smoke, training, and evaluation workflows.
  `docs/` holds decisions, paper assets, and the work log.
- `third_party/` contains pinned reference dependencies; avoid editing them for
  integrations. Keep datasets, checkpoints, logs, and generated output outside
  Git as specified by `.gitignore`.

## Build, Test, and Development Commands

- `pytest -q` runs unit tests; missing PyTorch skips them.
- `bash scripts/manifeel/check_environment.sh` reports revisions, paths, and
  CUDA/import status.
- `bash scripts/manifeel/setup_environment.sh` creates the ManiFeel Conda
  environment and installs editable references; it needs Conda, the licensed
  Isaac Gym archive, and a CUDA GPU.
- `bash scripts/manifeel/smoke_env.sh` checks headless Isaac Gym. Training and
  evaluation smokes need the dataset/checkpoint and GPU; use
  `MANIFEEL_*` overrides for paths.

## Environment, Network & Communication

- Use short, bounded preflight debugging before long training runs. Verify
  training parameters, batch size, CPU availability, DataLoader workers, GPU
  utilization, and checkpoint/log behavior during the preflight run.
- This server is for evaluation; large-scale training runs on `engram`, with
  code synchronized through GitHub. For local GPU evaluations, select only
  GPUs approved by the user, keep separate logs for concurrent processes,
  and use persistent output directories under `/data/wangzihao/`.
- Before starting GPU-dependent work, inspect the current GPU
  processes, report the GPU model, memory, utilization, and relevant running
  jobs to the user, and wait for confirmation before launching training or
  evaluation.
- A training process running only to keep a GPU busy may be terminated when
  the GPU is needed for the requested work; note the termination in the
  development log.
- Install persistent environments, caches, and artifacts under
  `/public/home/wangzihao/` (or the approved `/data/wangzihao/` data root), not
  `/tmp/`, which may be cleaned automatically.
- If a download is unusually slow or fails, first choose a domestic mirror or
  unset the proxy (for example, `MANIFEEL_UNSET_PROXY=1`). Do
  not try every route for every download; most ports work normally.
- If the approved GPUs are unavailable, do not replace a GPU-specific check
  with irrelevant CPU testing; report the limitation and wait for direction.
- Keep answers structured, evidence-based, and restrained. Recheck assumptions,
  state uncertainty, and request evidence when needed.

## Development Logs

- Read the relevant entries under `docs/logs/` before starting work, then record
  new development updates there instead of adding progress history to this
  file.
- Keep logs concise: record verified milestones, experiment commands and
  outcomes needed for reproducibility, and high-value pitfalls. Avoid routine
  narration and overly granular activity logs.

## Coding Style & Naming Conventions

Use four-space indentation, type hints, and public docstrings. Name functions,
variables, and files in `snake_case`, classes in `PascalCase`, and constants in
`UPPER_SNAKE_CASE`. Shell scripts should retain `set -euo pipefail`, source
`scripts/manifeel/common.sh` when applicable, and quote paths. No formatter or
linter is configured.

## Testing Guidelines

Name tests `test_*.py` and functions `test_*`. Prefer deterministic,
CPU-friendly tests for `adavip/`; keep GPU, licensed-artifact, and dataset
checks in ManiFeel smoke scripts. Run `pytest -q` and add regression coverage.

## Commit & Pull Request Guidelines

History contains only `Initial commit`, so no convention is established. Use
short, imperative subjects (for example, `Add progress-conditioned fusion`).
Pull requests should explain impact, validation, and environment assumptions;
link issues and include screenshots or logs for simulation or visual changes.
Call out submodule revisions and never include data, credentials, checkpoints,
or generated outputs.

## ManiFeel Review Notes

- Keep the Power Plug paper/baseline training series, including continuations
  and modality variants, at the upstream train and validation batch size of 8
  for comparable results. Do not apply the earlier throughput-oriented batch
  sizes of 512 or larger to this series unless the user explicitly requests a
  protocol change; make `batch8` explicit in new run and artifact names.
- ManiFeel demonstrations are Zarr trajectory stores. `shape_meta` selects the
  observations used by both the dataset loader and simulator: USB
  `vision_wrist` uses `wrist + state`; `vistac_wrist` and `visff_wrist` add the
  right TacRGB or force field. Recorded `wrist_2`, `front`, and `side` views
  are optional inputs, not evidence of multiple arms.
- Raw TacRGB and force-field inputs still pass through an encoder. The standard
  DP uses one learned ResNet18 per `type: rgb` key; the dedicated TacFF policy
  uses an MLP. UniT, T3, and AnyTouch are optional pretrained tactile branches
  and require separate representation weights and matching observation keys.
- A policy checkpoint must preserve the Hydra workspace/config and normalizer.
  Runtime policies implement `predict_action(obs_dict)` returning
  `[B, n_action_steps, action_dim]`; training also needs `set_normalizer()` and
  `compute_loss()`.
- **ManiFeel nine-task benchmark reporting:** use `test/mean_score`, **not**
  `test/success_rate`. The official runner averages each environment's maximum
  reward; its wrapper sets `reward = reset`. All nine pinned tasks set reset
  from `_check_success()` or timeout at 100000 steps, so in the 500-step runs
  `mean_score` equals the fraction of environments that reached simulator
  success. The local unit runner's additional `test/success_rate` uses only
  the final info window and can undercount; the official ManiFeel runner does
  not publish this field. Do not describe `mean_score` as merely a smoke
  signal or a bug for these runs, or use the old 10-env info-based results as
  paper-comparable scores. Canonical checkpoint values and protocol caveats:
  `docs/manifeel_benchmark_results.md`.
- The selected Power Plug row in that nine-task table is
  `checkpoints/0926/dp_power_plug_hyper_resnet_fusion_batch8_ep400_seed42/checkpoints/latest_epoch100.ckpt`,
  with `test/mean_score=0.66` (33/50). It uses `use_fusion=true` and
  `use_fusion_hypernet=false`, matching the eight other HyperResNet tasks;
  distinguish it from `checkpoints/0929ours/fusionhyper/latest_epoch300.ckpt`
  (extra fusion HyperNet, 0.56) and
  `checkpoints/batch8_tactile/Film_Fusion/latest_epoch200.ckpt` (CLIP/Sparsh
  FiLM, 0.08). Never mix these models' checkpoints or scores.
