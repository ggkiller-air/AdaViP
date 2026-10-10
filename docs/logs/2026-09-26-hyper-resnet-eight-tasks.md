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

## 2026-09-29 ModelScope Download and Evaluation

- With all proxy variables unset, downloaded `ggkiller/multi-fm@master`
  using `modelscope download --repo-type model --revision master
  --local-dir checkpoints --max-workers 5 ggkiller/multi-fm
  --include '0929ours/*/*.ckpt'`. All 27 checkpoints reside under
  `checkpoints/0929ours/<task>/latest_epoch{100,200,300}.ckpt`; the
  direct remote listing and local byte sizes match (143,055,999,067 bytes;
  no `.incomplete` files). Inventory: `remote_manifest.json` in the output
  directory below.
- Before evaluation, physical RTX 3090 GPUs 1/2/3/4/6 (24 GiB each) were
  idle at 4–6 MiB and 0% utilization. Occupied GPUs 0/5/7 were left alone.
  All nine tasks passed 10-environment, eight-step preflights. A prematurely
  terminated preflight left a stale `gymtorch` build lock; it was removed after
  confirming no compiler was running. Nut-bolt crashed on a preexisting
  zero-byte `factory_nut_m16_tight_subdiv_3x_R256_3c788ca8.sdf`; deleting
  only that corrupt cache entry allowed the original simulator configuration
  to regenerate it. Bulb installation also generated its SDF cache.
- Ran `MANIFEEL_0929_EVAL_PHASE=formal bash
  scripts/manifeel/eval_0929ours_checkpoints.sh` with saved EMA policies,
  10 environments, at most 500 simulator steps, seed 100000, and no videos.
  Object Search can finish before the step limit. All 27 `eval_log.json` files
  exist; each info-derived, episode-ever `test/success_rate` matches its 10
  individual `test/sim_success_*` outcomes (270 checked). This diagnostic
  field is not the ManiFeel benchmark score: under these tasks' 500-step
  horizon, `test/mean_score` is the fraction of simulator successes. The
  epochs share an evaluation seed.

  | Task | Epoch 100 | Epoch 200 | Epoch 300 |
  | --- | ---: | ---: | ---: |
  | Ball Sorting | 5/10 | 5/10 | 4/10 |
  | Bulb Installation | 6/10 | 10/10 | 9/10 |
  | FusionHyper (Power Plug) | 5/10 | 5/10 | 5/10 |
  | Gear Assembly | 6/10 | 5/10 | 4/10 |
  | Nut-Bolt Assembly | 7/10 | 9/10 | 6/10 |
  | Object Search | 10/10 | 10/10 | 10/10 |
  | Peg Insertion | 5/10 | 3/10 | 1/10 |
  | Peg Reorientation | 5/10 | 3/10 | 6/10 |
  | USB Insertion | 6/10 | 6/10 | 6/10 |

- Outputs, process logs, `success_rates.csv`, and `summary.md`:
  `/data/wangzihao/outputs/manifeel/eval/0929ours_batch8_n10_steps500_seed100000_20260929/`.
  Focused tests: 11 passed. Full `pytest -q`: 78 passed, 1 skipped, and
  3 unrelated training-launcher tests failed because local `pih_quan_June06`
  and `sorting_quan_Aug8` datasets are absent; evaluation did not need them.

- Combined the nine 0929ours rows with the previously evaluated Power Plug
  ResNet Fusion baseline (`use_fusion=true`, `use_fusion_hypernet=false`) from
  the 0926 series, whose epoch-100/200/300 SRs are 9/10, 8/10, and 8/10.
  The new fourth score averages the three epoch SRs within each row, not
  across tasks. Verified the baseline's three `eval_log.json` files and all
  10 per-environment outcomes per epoch. Combined tables:
  `combined_0926_0929_summary.md` and `combined_0926_0929_success_rates.csv`
  in the 0929 evaluation output directory above.

## 2026-09-29 Best Epochs Under the Original Evaluator

- Selected the highest previous **10-environment info-derived diagnostic SR** among
  epochs 100/200/300 for each of the ten rows above, breaking ties toward the
  later epoch: Power Plug fusion-only 100, Power Plug fusion-HyperNet 300,
  Peg Insertion 100, USB Insertion 300, Gear Assembly 100, Nut-Bolt Assembly
  200, Bulb Installation 200, Peg Reorientation 300, Object Search 300,
  Ball Sorting 200. Selection and evaluation reuse seed 100000, so this is a
  selected-checkpoint comparison, not an independent held-out model selection.
- Ran the pinned `third_party/manifeel/eval.py` and its saved unit runner
  **without editing either**. The launch-only shim
  `scripts/manifeel/eval_upstream_checkpoint.py` places checkpoint loading on
  CPU and points the policy, simulator and graphics at the same approved
  physical RTX 3090; CUDA/Vulkan ordinals are not remapped. GPUs 1/2/3/4/6
  were idle before evaluation and were used; occupied GPUs 0/5/7 were left
  alone. The first Peg Insertion preflight incorrectly targeted GPU 0 because
  its config is named `isaacgym_config.yaml`; fixed the shim's device matching
  and reran it on GPU 3. All 10 preflights completed with 50 environments,
  8 steps and 2 videos.
- Formal evaluation used 50 environments, at most 500 simulator steps,
  seed 100000, the saved EMA policy, and 2 videos per run. Original Isaac Gym
  consistently exited 139 during teardown *after* writing `eval_log.json`;
  accepted only runs with a valid result, checked their GPU, seed and
  environment-count logs, and verified both saved videos for each of the 10
  results. Outputs and complete logs:
  `/data/wangzihao/outputs/manifeel/eval/original_batch8_best_epochs_n50_steps500_seed100000_20260929/`
  (`summary.md`, `results.csv`, `formal/`, `logs/`).

  | Task | Best epoch | Original `test/mean_score` | Original `test/success_rate` |
  | --- | ---: | ---: | ---: |
  | Power Plug fusion-only | 100 | 66% | 58% |
  | Power Plug fusion-HyperNet | 300 | 56% | 44% |
  | Peg Insertion | 100 | 34% | 34% |
  | USB Insertion | 300 | 56% | 54% |
  | Gear Assembly | 100 | 60% | 60% |
  | Nut-Bolt Assembly | 200 | 88% | 88% |
  | Bulb Installation | 200 | 80% | 38% |
  | Peg Reorientation | 300 | 42% | 42% |
  | Object Search | 300 | 100% | 82% |
  | Ball Sorting | 200 | 58% | 50% |

- `test/mean_score` is the benchmark **fraction of 50 environments reaching
  simulator success** in these runs: the reset-derived reward cannot include
  task timeouts within 500 steps (configured timeout: 100000). The modified
  local unit runner's additional `test/success_rate` retains only the last
  observation window of simulator info; it is not the benchmark score. The earlier
  10-environment info-based diagnostic used fewer environments and must not
  replace the formal results.

## 2026-10-10 Nine-task diagnostic plan

- Audited the ten-environment info-based diagnostics and the original
  50-environment evaluation against ManiFeel Table II; documented the metric
  distinction and experimental protocol differences in
  `docs/manifeel_benchmark_results.md`, which also contains the investigation
  steps. Peg Insertion's info-based diagnostic
  lost four successes (and gained none) between epochs 100 and 300 on the
  same ten seed indices.
- Added evaluation-only attention recording, an optional labeled-stage
  modality heatmap, and success-stratified executed-action summaries. The
  earlier nine-task outputs did not include attention or action traces; no
  simulator evaluation or training was started. Focused CPU tests: 16 passed;
  full `pytest -q`: 82 passed, 1 skipped, 3 pre-existing launcher dry-run
  failures due to missing `pih_quan_June06` and `sorting_quan_Aug8` datasets.
- Before any new simulator work, `nvidia-smi` reported eight RTX 3090
  (24 GiB) devices. Physical GPUs 3/4/6 showed 2 MiB and 0% utilization;
  0/1/2/5/7 had compute processes, with GPU 1/2 at 90%/68% utilization.
  No GPU has been approved for this diagnostic evaluation yet.

## 2026-10-10 Metric clarification

- Rechecked the **official** ManiFeel runner: its benchmark output is
  `test/mean_score = mean(max(reward) by environment)`; the environment
  wrapper explicitly substitutes `reward = reset`. All nine pinned tasks set
  that reset flag from `_check_success()` or the configured 100000-step task
  timeout; the 500-step formal runs cannot reach that timeout. Thus the saved
  `mean_score` is the benchmark fraction of simulator successes in these
  runs, whereas the extra `test/success_rate` is final-window-only. The JSON
  contains aggregate metrics and two videos, not all per-step success traces;
  a GPU rerun is needed only for detailed timing diagnostics.

## 2026-10-10 Power Plug baseline cross-check

- Located `checkpoints/batch8_tactile/baseline/latest_epoch{100,200,300}.ckpt`
  and the existing 50-environment corrected evaluation at
  `/data/wangzihao/outputs/manifeel/eval/batch8_power_plug_5methods_n50_steps500_seed100000_20260926/formal/baseline/`.
  Its benchmark `test/mean_score` and episode-ever `_check_success` both equal
  25/50, 32/50, and 29/50, respectively; independently compared all 150
  saved per-environment `sim_max_reward_*` / `sim_success_*` pairs, with zero
  disagreements. The three-checkpoint average is 57.33%, compared with the
  ManiFeel TacRGB Power Plug Table II value of 51% (three training seeds and
  last ten epochs). Epoch 100 alone is 50%, but this is not a full reproduction
  of the paper aggregate.
- Verified pinned Power Plug task's reset buffer uses `_check_success` or task
  timeout (`max_episode_length=100000`); 500-step evaluation cannot hit that
  configured task timeout. `mean_score` is the correct reporting field for
  this baseline and the nine formal results below. No new GPU run was started.

## 2026-10-10 Nine-task metric and protocol resolution

- Checked all nine pinned simulator task `_update_reset_buf` implementations
  and nine task YAMLs: `reset_buf` is `_check_success()` or a timeout at
  `max_episode_length=100000`. Under the saved 500-step evaluation, the
  timeout cannot trigger; the official-style `test/mean_score` therefore
  measures the fraction of environments reaching simulator task success at
  least once. The extra `test/success_rate` records only the last info window
  and is not the benchmark score. This resolves the earlier cautious notes
  above for these nine particular runs without claiming per-environment
  trajectories exist in their aggregate JSONs.
- Independently parsed all nine selected checkpoints' formal 50-environment
  `eval_log.json` files: each `test/mean_score` agrees with `results.csv`,
  each is an integer multiple of 1/50, and each run log shows 50 environments
  and seed 100000. Values (peg, USB, plug, gear, nut-bolt, bulb,
  reorientation, search, sorting) are 17, 28, 28, 30, 44, 40, 21, 50, and
  29 successes out of 50, respectively. No new simulator run was started.
- Protocol caveat for comparisons to ManiFeel Table II: our eight-task
  launcher forces `vistac_wrist` and 50 demos for all tasks; paper Table VI
  uses front view for three exploration tasks, 30/20 demos for nut-bolt/bulb,
  and lists 6D actions for nut-bolt/bulb versus our 7D configuration. The
  paper separately measures substantial wrist/front effects on object search;
  do not present the descriptive differences as fully matched experiments.
  The canonical reporting rule and checkpoint table are in
  `docs/manifeel_benchmark_results.md`.

## 2026-10-10 Documentation consolidation

- Added `docs/manifeel_benchmark_results.md` as the canonical source for the
  metric decision, nine checkpoint paths, verified 50-environment scores, and
  the consolidated single-task diagnostic and improvement plan;
  `AGENTS.md` now points to it and identifies this machine as the evaluation
  host. Replaced contradictory `mean_score`/final-window success statements
  in the retained logs. Removed obsolete `docs/manifeel_experiments.md` (old
  batch-512/Slurm recipe); preserved the paper assets and reproducibility
  logs. The separate improvement-plan file was consolidated into the
  canonical document and removed. No GPU work was started.
- Translated the canonical document into Chinese in place; checkpoint paths,
  score definitions, numeric results, evaluation commands, and the single-task
  improvement plan remain together under the same filename. No new GPU run.
- Resolved the Power Plug model identity in the nine-task table: the selected
  model is `checkpoints/0926/dp_power_plug_hyper_resnet_fusion_batch8_ep400_seed42/checkpoints/latest_epoch100.ckpt`
  (`use_fusion=true`, `use_fusion_hypernet=false`), not the unrelated
  `0929ours/fusionhyper` or `batch8_tactile/Film_Fusion` models. Its existing
  50-environment, seed-100000 result is `test/mean_score=0.66` (33/50):
  15 percentage points above the paper's TacRGB Power Plug 51% mean, as a
  descriptive comparison only. The FiLM checkpoint remains an alternate 8%
  (4/50) result, not the current table entry. The other eight values did not
  change. Verified all 48 checkpoint files and the ManiFeel Table II TacRGB
  comparison in `docs/manifeel_benchmark_results.md`; no GPU run or file move.

## 2026-10-10 Object Search offline audit

- Original 50-environment evaluation at seed 100000 records `test/mean_score=1.0`, last-info-window `test/success_rate=0.82`; 10-environment epoch 100/200/300 evaluations also score 1.0 but reuse the same seed. Original 2 videos do not represent all 50 environments; contact sheet saved under `/data/wangzihao/outputs/manifeel/eval/object_search_audit_20261010/`.
- Pinned Object Search insertion task calls `_check_plug_close_to_socket()` for success: average plug–socket keypoint distance must be strictly within 0.12–0.20 m. No grasp/classification condition; socket/plug positional noise is zero, plug rotation noise nonzero. The metric is correct for this simulator predicate, but independent semantic completion and policy dependence remain unverified. Paper Table VI uses **front** vision for its Table II 69% comparison; Fig. 13 reports 92.7% for wrist visuotactile policies.
- GPU inventory before proposed follow-up: RTX 3090 ×8; GPUs 3/4/6 idle (2 MiB, 0% utilization), GPUs 0/1/2/5/7 occupied. Reported availability to user; no GPU job launched because specific device confirmation is still pending. Next run: 50 env, new test seed, diagnostic trace plus zero-action control on a user-approved physical GPU.
