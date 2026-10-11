# No-fusion zero-modulation evaluation (2026-10-11)

- User requested inference γ/β zeroing in the existing HyperNet/no-fusion
  epoch300 checkpoint and authorized GPU3/4. User additionally waived repeated
  GPU permission requests on this evaluation server; continue inspecting
  allocations and leave unrelated jobs untouched. No local training authorized.
- Read prior mechanism/probe logs. Snapshot02:34UTC: RTX3090 GPUs3/4/6 each
  2MiB/0%, no compute allocation; GPUs0/1/2/5/7 retain existing jobs. Cgroup
  memory limit64GiB, current~16.2GiB, memory PSI0. Keep at most two evaluations.
- Planned matched evaluation seeds300000/400000,50 environments,500 steps,
  official mean_score; same saved EMA, normalizer and100 inference steps.
  Per-GPU8-step zero/trace preflight verifies exact zero modulation and
  unchanged features; excluded from official counts. Outputs/commands:
  `results/no_fusion_zero_batch8_20261011/`. Reuse existing normal/no-fusion
  and baseline runs for paired comparisons; no dataset download/retraining.
- Both50-env8-step preflights passed: generated gamma/beta exactly zero,
  modulated features bitwise equal raw features, no attention recorded, binary
  rewards and logged mean_score agree. Full500-step evaluations started on
  GPU3/seed300000 at02:36:31UTC and GPU4/seed400000 at02:36:34UTC. RootAGENTS
  now reflects the user's standing evaluation-GPU authorization; no local
  training and no interference with unrelated jobs remain in force.
- Both full evaluations completed: zero seed30000029/50=.58, seed40000029/50=.58,
  pooled58%; matched normal no-fusion64%, baseline54%. Zero-minus-normal−6pp,
  conditional paired environment bootstrap95%[−14,+2], gained6/lost12,
  unadjusted sign p=.237885. Zero-minus-baseline+4pp,[−7,+15], gained18/lost14,
  p=.596615. Neither necessity, superiority nor equivalence is established.
- All12600 formal encoder frames: gamma/beta exactly zero, modulated features
  bitwise equal raw features, encoder output bitwise equal raw tactile/vision
  and normalized-state concatenation, no attention. Alpha preserved. All50
  per-environment logged maxima match binary rewards for each seed; all five
  initial-condition arrays match normal/baseline exactly within each seed.
- Reports/figure/intervention audit and exact commands under result root;
  PNG layout inspected. Probe/helper tests28 passed; shell syntax and tracked
  whitespace checks passed. No training/data download; artifacts~141MiB.
  Own evaluations exited, GPU3/4 returned2MiB/0%, unrelated jobs retained.
