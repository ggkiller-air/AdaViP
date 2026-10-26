# Main: portable Stage2 evaluation (2026-10-26)

- Created `/public/home/wangzihao/projects/AdaViP-main` on `main`, advanced to
  `origin/main` (`7660250`), then merged `Stage2_Concat_T2_utility` (`60be99c`)
  and `experiment/stage2_mlp_fusion` (`1e2ec7f`). Preserved both model classes,
  training entrypoints and Normalizer restore fixes. The three existing training
  worktrees and their uncommitted files were left unchanged. T6 remains on its
  existing branch/worktree.
- Extended `eval_full_sensor_checkpoint` to identify legacy Stage2 fusion from
  checkpoint tensors, strictly restore EMA (or explicitly requested raw model),
  RMS and Normalizer, and retain Stage0 support. Extracted the inference
  architecture from the actual shared Teacher E500 published config into
  `configs/manifeel/eval_stage2_t2.yaml`; no Teacher weights, Utility label store,
  dataset, optimizer construction or decoded RAM cache is needed for evaluation.
  The frozen task embedding archive is still an explicit portable input.
- Added a sequential Transformer/MLP checkpoint batch evaluator with per-run
  logs, incremental `summary.json`, upfront missing-file checks, and fail-on-error
  behavior. Added `--check-only` for strict CPU restoration without importing
  Isaac Gym or starting a simulator. Usage and 4090 camera preflight are in
  `docs/stage2_evaluation.md`.
- CPU verification: the initial targeted policy/evaluation suite passed 20
  tests, including numerical action parity after restoring both model variants
  with raw and EMA weights. Both actual archived job 387293 / 387295 epoch20
  ckpts strictly loaded with the default inference config and all saved
  Normalizer tensors. Reports/logs are outside Git at
  `/data/wangzihao/outputs/manifeel/stage2_main_cpu_check_20261026/`.
- The first full CPU suite reported 232 passed, 1 skipped, 5 failed. Four
  failures were caused by initially empty submodule directories in the new
  worktree; local read-only reference clones were checked out at their pinned
  gitlink revisions; all four checks then passed. The final targeted run passed
  25 tests, including the new batch settings/aggregation regression.
- Per the user's explicit request, removed the unused legacy
  `test_film_checkpoint_uses_local_assets`, whose dependence on installed
  `third_party/CLIP` / Sparsh source assets repeatedly failed this environment.
  Runtime FiLM asset validation remains intact. No RTX/GPU evaluation or training
  was launched.
- Final full CPU suite after removing that obsolete test: **237 passed,
  1 skipped**, with five existing PyTorch Transformer warnings. Shell syntax,
  Python compilation, CLI help and `git diff --check` also passed.
