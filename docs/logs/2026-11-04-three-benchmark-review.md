# ManiFeel / Tabero / UniVTAC code and method review

Reviewed the latest local changes and the active immutable-run contracts on
2026-11-04. Evidence snapshot: `/data/wangzihao/outputs/review_20261104_status.json`
(21:49 CST). No jobs were cancelled, submitted or restarted; active source
snapshots, weights, labels and optimizer state were not modified.

## Findings and changes

- Fixed progress reporting: a training contract alone previously counted as
  training although it precedes preflight/optimization. It now reports
  initialization until a real update event or checkpoint exists, retains
  training across queued continuations, and recognizes both completion-marker
  spellings. Added regression coverage and fixed summary-only registry lookup.
- Fixed reproducibility gaps: the checked-in Tabero Stage2 config omitted
  required `cache_id`, and UniVTAC's selected E300 Student had no checked-in
  launch template. Added the exact cache ID and a UniVTAC config template;
  run-specific source identity and archive destination remain launch metadata.
- Fixed `.gitignore`: the broad `logs/` rule silently excluded new development
  records under `docs/logs/`. Allow Markdown development records while retaining
  the generated-log exclusions. The newly published records also clarify the
  final protocol, superseding the November 2 architecture proposal.
- Protocol limitation, preserved for checkpoint compatibility: both Tabero DP
  and the pinned native VTLA tactile encoder stack two causal three-tap blocks
  and pool the last output. Their receptive field is five frames, despite
  storing reference plus eight history frames. A direct gradient probe gave
  exactly zero for input frames 0–3 and nonzero gradients for frames 4–8.
  Documented and pinned this behavior in a regression test. It is shared with
  the native implementation, not a DP port discrepancy. Using the full history
  would require a separately versioned Teacher, regenerated labels and Student;
  silently changing the current encoder would invalidate the selected weights.

## Method assessment

- DP uses each benchmark's selected no-dropout, late-language L1 Teacher:
  ManiFeel E50 EMA, UniVTAC six-task E300 EMA, Tabero E150 EMA. Encoders and
  normalizers are inherited; the DP action U-Net is fresh. Utility receives a
  detached context, and action conditioning receives detached predicted utility.
  A trains utility; B adds the actor; C unfreezes inherited encoders.
- Native FTP-1 and Tabero-VTLA reuse released action models with observation
  FiLM and action-expert LoRA. There is no added query/attention/action residual.
  Their approved C phase keeps the B parameter whitelist and reduces LR; it is
  not DP-style encoder unfreezing. This matches the November 3 final protocol.
- Paired full/removal comparisons share noise and diffusion/flow time and keep
  signed differences. Labels preserve their native loss semantics: UniVTAC DP
  8D, Tabero DP weighted 13D, FTP masked 120D container, and native Tabero weighted
  32D including 19 padding channels. Native Tabero padding loss is intentional
  baseline compatibility, but its utility can reflect padding prediction as
  well as physical action prediction. Compare physical position/force components
  separately in analysis; do not compare raw utility magnitude across backbones.
- DP uses K64 random stratified draws with variance estimates; native models use
  K8 inverse-CDF midpoint quadrature plus sampled noise. The latter does not
  store independent replicas or variance. It is a lower-budget approximation,
  not equivalent label precision. Retain signed labels, including frequent
  negative wrist-camera values for some FTP tasks; do not clip them away.
- No-dropout Teacher ablations are off the training input distribution. Utility
  measures that fixed model's sensitivity to removal under the specified loss;
  it does not establish causal sensor value or improved closed-loop success.
  C-phase DP encoder changes also make utility prediction track a moving feature
  representation against fixed Teacher targets. Held-out task metrics and
  closed-loop ablations remain necessary before claiming a method improvement.
- Read-only K64 uncertainty audit: the fraction with `abs(u) <= 1.96*SE` is
  0–3.10% per dimension for Tabero, and 0–11.80% for UniVTAC (largest for T2).
  This is a normal-approximation diagnostic of Monte Carlo sign uncertainty,
  not a calibrated confidence guarantee. All examined arrays were finite.
  UniVTAC's full-window policy uses 62,540 of 67,040 native anchors; Tabero
  retains all 109,097 anchors with episode padding masks. Preserve this distinction.

## Training snapshot, 21:49 CST

All ten jobs were RUNNING, Account=yangyc, QOS=user_wangzihao, Requeue=1.
Metrics advanced during review; recent losses were finite. Recent allocation
telemetry identifies H200 GPUs with 143771 MiB total and 25.3–36.3% memory used,
including the separate 16% reservation. No new GPU computation was launched by
this review. Single utilization samples vary; progress logs establish liveness.

| Route | Job | Verified progress |
|---|---|---|
| ManiFeel DP | 423348 | C, E155/600 in progress, step314460; 65,265 K64 labels complete |
| Tabero DP | 426706 | C, E51/200 in progress, step85980/341000; 109,097 K64 labels complete |
| UniVTAC DP | 426602 | C, E17/600 in progress, step33000; 62,540 K64 labels complete |
| FTP insert_hole | 426113 | C, 16220/20000 updates |
| FTP insert_tube | 426114 | C, 18740/20000 updates |
| FTP lift_bottle | 426115 | C, 16200/20000 updates |
| FTP lift_can | 426116 | C, 17440/20000 updates |
| FTP pull_out_key | 426117 | C, 18140/20000 updates |
| FTP put_bottle_in_shelf | 426118 | C, 11120/20000 updates |
| Tabero-VTLA | 426119 | K8 labels 70568/109097 (64.7%); formal Student not started |

All six FTP label sets are complete. DP and FTP latest checkpoints and retained
periodic snapshots exist. Native Tabero's successful six-update trainer smoke
is diagnostic only. No new Student closed-loop score was found in the registry.
Historical ManiFeel HyperResNet scores and the user-reported Tabero Teacher
RTX rollouts must not be presented as these Students' results.

## Validation

- Full AdaViP regression before review fixes: 397 passed, 2 skipped, 5 warnings,
  226.24 seconds, using the declared ManiFeel environment and dependency paths.
- After fixes: 25 focused tests passed in 19.35 seconds, covering Tabero native
  loss, labels, model gradients, exact A/B/C resume, progress states and TCN
  receptive field. Python/shell syntax and whitespace checks run before commit.
- UniVTAC `scripts/check_l1_uvip.py`: all ten checks passed, including native
  paired loss equality, microbatch independence, inherited context, identity
  FiLM, fresh actor, phase freezing, gradient isolation, inference, six-task
  filtering and exact raw/EMA/Adam/scheduler resume across A/B/C.
- These are local CPU regressions plus inspection of existing actual GPU
  preflight receipts; they are not a new simulator rollout or GPU retraining.
- Test logs: `/data/wangzihao/outputs/review_20261104_adavip_pytest.log`,
  `review_20261104_fixes_pytest.log`, and `review_20261104_univtac.log`.
