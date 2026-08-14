# Repository Guidelines

## Project Structure & Module Organization

- `adavip/model/` contains adaptive perception and transforms; `adavip/policy/`
  wraps them around an action backbone.
- `tests/` contains PyTorch/pytest tests; `configs/` stores experiment settings.
- `scripts/manifeel/` and `slurm/manifeel/` provide local and cluster smoke
  tests. `docs/` holds decisions, paper assets, and the work log.
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

- Use interactive GPU allocations for short, bounded preflight debugging
  before long training runs. Verify training parameters, batch size, CPU
  allocation, DataLoader workers, GPU utilization, and checkpoint/log behavior
  during the preflight run.
- Submit long-running training and formal evaluation jobs through `sbatch`.
  Before submission, use the interactive-GPU preflight results to request
  enough CPUs, set a sufficiently large batch size when memory allows, and
  increase DataLoader workers to keep the GPU utilized. The cluster has ample
  CPU capacity, so avoid under-requesting CPUs for GPU jobs.
- Before starting GPU-dependent work, inspect the current allocation and GPU
  processes, report the GPU model, memory, utilization, and relevant running
  jobs to the user, and wait for confirmation before launching training or
  evaluation.
- A training process already running only to keep an interactive allocation
  alive may be terminated when the GPU is needed for the requested work; note
  the termination in the development log.
- Install persistent environments, caches, and artifacts under
  `/public/home/wangzihao/` (or the approved `/data/wangzihao/` data root), not
  `/tmp/`, which may be cleaned automatically.
- If a download is unusually slow or fails, first choose a domestic mirror or
  unset the proxy (for example, `MANIFEEL_UNSET_PROXY=1`). Do
  not try every route for every download; most ports work normally.
- GPU unavailability is normal on login or CPU nodes. Do not replace a
  GPU-specific check with irrelevant CPU testing; report the limitation and
  ask the user to request an interactive GPU or submit an `sbatch` job.
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
- `test/mean_score` currently aggregates the maximum reset-derived reward.
  Treat it as a smoke signal until a verified task success signal such as
  `info["successes"]` is logged and used for benchmark comparisons.
