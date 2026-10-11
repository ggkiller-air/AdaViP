# Power Plug GPU mechanism probes (2026-10-10)

- Continued the CPU demo probes after reading the earlier log. User requested
  Power Plug only and all outputs under repository `results/`; moved the original
  output tree to `results/hypernet_batch8_20261010/` and ignored `results/` in Git.
  `results/README.md` indexes figures, measured findings and limitations.
- User explicitly authorized shared GPU5 and idle GPUs3/4/6. Preflight snapshots
  identified RTX3090, 24576MiB each; GPUs3/4/6 had2MiB/0%, GPU5 retained an existing
  ~9339MiB job. Used no other GPUs, killed no process, launched no training.
  GPU5 policy preflight preserved the saved EMA/normalizer/100-step scheduler,
  produced [1,8,6] actions, used ~1365MiB peak, and repeated exactly.
- GPU5: paired-noise demo action sensitivity, demos0/1/2, 8 histories per demo,
  seeds11/22/33, batch8. Mean modulation calibrated on disjoint demos3/4/5.
  Main/baseline vision-blur RMS .189153/.175710, tactile-blur .013986/.026947;
  exact repeat RMS0. Spatial4x4 patch-blur maps on demo1 frames33/102/170 use
  kernel31/sigma10 in both history frames. Added CPU-only comparison with shared
  heatmap color limits and exact input/scale/seed matching checks.
- GPUs3/4/6: baseline/main/extra-fusion-HyperNet saved EMA, 50 environments,
  500 steps, seed100000. Reproduced test/mean_score .50/.66/.56 (25/33/28 successes).
  All initial state and plug/socket pose max errors0. Main versus baseline:
  22 both succeed, 11 main-only, 3 baseline-only,14 both fail. This reuses the
  known evaluation seed, not an independent-seed replication. Paths and commands
  are in `docs/manifeel_hypernet_probes.md`.
- Recorded pairwise plug/socket force, task poses, per-step rewards and explicit
  [chunk,env,history,...] modulation axes; excluded inputs on/after first benchmark
  success. First-contact threshold .001; contacted counts stable at1e-4/1e-3/1e-2.
  Main attention near[1,1,0,1] before and after contact. Environment-level averaging
  and environment bootstrap avoid frame pseudoreplication. No evidence for a
  simple post-contact tactile-attention increase.
- GPUs4/3/6: main-checkpoint inference interventions none/mean/zero, 10 environments,
  500 steps, new seed200000. Scores .70/.80/.80; initial states exactly matched.
  Both interventions add the same single success. This small diagnostic does not
  establish dynamic modulation utility or retrained-architecture superiority.
  `controlled_rollout_report/` stores paired outcomes and figures.
- GPU5: modulation-only actions from cached rollout conditions, first20 eligible
  environments, one pre/post-contact history each,3 seeds, batch8. Mean γ/β RMS
  .004186/.002704; zero .016095/.023242; cross-phase .004092/.006054.
  Reconstructed encoder max error2.98e-7. Initial saved-condition baseline failed
  exact-repeat control because attention rebatching introduced small differences
  amplified by diffusion. Retained that failed attempt but excluded it from all
  results. Successful rerun uses dynamic reconstructed baseline and bitwise exact
  repeats, not a loosened tolerance. Neither action RMS nor attention proves utility.
- Validation: full CPU-hidden suite104 passed,1 skipped,3 existing failures for
  missing pih_quan_June06/sorting_quan_Aug8. Output:
  `results/hypernet_batch8_20261010/pytest_full_gpu_probes_final.log`.
  Focused probe/diagnostics/attention/comparison suite32 passed; source compilation
  and tracked/untracked patch whitespace checks passed.
  No unrelated dataset/launcher failure was modified. Final GPU check: own probes
  finished, GPUs3/4/6 back to2MiB/0%; GPU5 original9339MiB allocation untouched.
- Remaining: multi-seed larger closed-loop interventions, contact-phase-restricted
  interventions and matched-protocol retraining ablations (training on engram).
