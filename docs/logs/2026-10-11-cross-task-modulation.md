# Cross-task modulation evaluation (2026-10-11)

- User requested broader simulation evidence after Power Plug probes. Earlier
  single-task scope followed user preference; now preselect two additional
  correct-camera tasks before seeing results: Peg Insertion epoch100 and
  Nut-Bolt Assembly epoch200 from canonical nine-task checkpoint table.
- Read benchmark, eight-task training and recent mechanism logs. Selected
  checkpoints exist locally; no independent non-Power baseline found in local
  checkpoint inventory. Paper scores are not a matched local baseline.
- Plan normal/zero vision+tactile gamma/beta across seeds300000/400000,50envs,
  500steps, preserved EMA/normalizer/100 inference steps and fusion:8 formal runs,
  per-task100 shared initial conditions. Exclude known wrong-camera tasks.
- 03:54UTC resource snapshot: GPUs3/4/6 RTX3090 24GiB,2MiB/0%, no allocation;
  other GPUs retained jobs. Use3/4, at most2 concurrent; cgroup64GiB,current~21.3GiB,
  PSI0. Standing GPU authorization; no training, new data or checkpoint downloads.
  Audit config and8-step zero traces before full evaluations. Output:
  `results/cross_task_zero_batch8_20261011/`. Do not promise a favored mechanism
  merely because performance improves; report every preselected outcome.
- CPU audit confirms both saved encoders are HyperResNet+fusion, no extra
  fusion HyperNet; EMA,train/val batch8,training seed42,50 demos and100 diffusion
  steps. Peg6D actions,nut-bolt7D; both2 observation histories/8 action steps.
  Runtime Hydra configs use TacSLTaskInsertion and TacSLTaskBoltNut with
 100000-step task timeout, wrist input selected by shape_meta. The separate
  IsaacGym repository's default task files are not the runner's Hydra configs.
- Both8-step zero preflights passed exact-zero/raw-feature/reward/log checks,
  with fusion attention preserved. Formal normal300000 runs started03:59:28UTC
  on3 for Peg,03:59:54UTC on4 for nut-bolt. Memory PSI remains0.
- All8 formal50-env/500-step evaluations completed: Peg normal .42/.30,
  zero .42/.26, pooled36/34%; nut-bolt normal .88/.76, zero .86/.82, pooled82/84%.
  Peg zero-minus-normal−2pp, conditional95%[−9,+5], gained5/lost7,p=.774414;
  nut-bolt+2pp,[−7,+12], gained13/lost11,p=.838820. These are unadjusted,
  exploratory, conditional environments; no equivalence/training-seed claim.
- Every recorded reward is binary; all50 per-env maxima match official logs
  in every run. All five initial-state/pose arrays match within task/seed.
  Four zero runs record25200 encoder frames: exact zero gamma/beta and raw
  features, fusion remains. This is200 shared new task-specific initial
  conditions, not400 independent starts from8 evaluations.
- Normal pre-success traces: Peg attention nearly all vision, fusion/vision
  norm~.874; nut-bolt tactile head means~.85–.87 but fusion/vision~5.6e-7,
  tanh gate4.65e-6. Direct tactile concat persists. Stage groups differ in
  eligibility (Peg100/86, nut100/100); no causal phase-switch inference.
- Cross-task PNG/PDF/JSON, separate paired reports, descriptive normal traces,
  runtime wrist-camera/hash audit, exact commands and original traces saved.
  Figures inspected;31 related helper tests passed; scripts compile/shell
  syntax/tracked whitespace clean. No new training/data/checkpoint download;
  outputs~671MiB. Final04:38:48UTC snapshot: GPUs3/4 returned2MiB/0%, other jobs
  retained; memory PSI0 throughout monitored runs. Canonical nine-task table
  unchanged; no matched local non-Power baseline added or claimed.
