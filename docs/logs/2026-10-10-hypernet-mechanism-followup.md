# HyperNet mechanism follow-up (2026-10-10)

## Score recheck (2026-10-11)

- User queried whether baseline54% uses the same mean_score metric. CPU-only
  recheck of four existing complete raw500x50 binary reward arrays reproduced
  baseline32/50 and22/50, main36/50 and30/50. All50 per-environment logged maxima
  match raw rewards in each run; aggregate means match `test/mean_score`.
- Baseline/main initial states and plug/socket positions/quaternions match
  exactly within each evaluation seed300000/400000; all four logs show no
  intervention or policy ablation. Shared device runner inherits official
  `ManifeelRunner.run`, computing mean of per-environment maximum rewards.
  Audit: `results/mechanism_followup_batch8_20261010/baseline_score_audit_20261011.json`.
  Baseline54% pools new seeds, not the old seed100000 result64%. No GPU work.

- User requested cause exploration, allowed at most1–2 additional task datasets
  to conserve disk, allowed idle GPUs, and prohibited local training. Read prior
  Power Plug logs and inspected GPU/process/disk state. RTX3090 GPUs3/4/6 were
  2MiB/0%; 0/1/2/5/7 retained jobs. Reported snapshot and requested confirmation
  per repository rule. No new GPU evaluation or training launched while waiting.
- Results under `results/mechanism_followup_batch8_20261010/`; README indexes
  verified findings, caveats, commands and planned paired-seed evaluations.
- Reused prior formal50-env/500-step/seed100000 records; verified50 individual
  max rewards match each score. Baseline100/200/300=.50/.64/.58; main=.66/.66/.62;
  HyperNet-no-fusion=.60/.66/.70; static-fusion=.52/.54/.58. Same-epoch main gains
  are16/2/4pp; selected maximum only2pp, still selection-biased on reused seed.
  Audited5 saved checkpoints on CPU: seed42, train/val batch8,50 demos,val_ratio.02,
 100-step inference, identical normalizer tensors. Weight distance is not utility.
- CPU replay of primary rollout context on1563 same-time/same-phase non-success
  donor pairs: exact repeat0, gamma/beta max replay error2.06e-6. Tactile-context
  swaps change fused condition .00525 pre/.01314 post; vision .01258/.01250;
  state .00082/.00047. These are condition deltas, not actions or success effects;
  cross-phase perturbation magnitudes are not standardized.
- Real Q/K score replay: Power Plug error3.96e-9, all heads max probability≥.99
  on all1563 inputs; signed logit median~17.11/14.56/−14.02/16.93. Tanh modulation
  abs≥.95 fractions~.795/.877/.921/.899 for vision gamma/beta and tactile gamma/beta.
  Confirmed saturation, not an explanation of how training reached it. Direct
  tactile concatenation remains, and small fusion gate does not imply small branch.
- Geometric audit of matched baseline100/main100: failure buckets no-contact6/4,
  contacted-far15/8, contacted-near4/5, success25/33, using2x pinned success error
  threshold .0069916m. Eight of11 gained successes were baseline far failures;
  all3 lost successes became main near failures. Tested1.5/2/3/4x thresholds;
  classifications are descriptive, not inferred causal stage labels.
- Downloaded only official pih_quan_June06.zip with proxy unset via existing
  range downloader (8 workers,2 retries), verified assembled size4704135973.
  Extracted first6 complete episodes (922 frames), wrist/right TacRGB/state/action/
  force-field; retained~252MiB under approved /data root. Checked every needed chunk
  exists and zip CRC passes when read; truncated Zarr shapes/episode ends, decoded
  endpoints, then successful CPU8-frame preflight and96-frame demo probe. Kept
  source manifest/SHA256 in results; removed only this run's new archive and parts.
  Original Power Plug archive/data/checkpoints untouched; no second task downloaded.
- Peg attention replay error1.19e-7, tactile weights~.000016/.000936/.000015/.000020;
  signed score medians~−12.61/−9.14/−12.60/−12.30. Demonstration-only exploration;
  no Peg benchmark measured. Candidate next claim is cross-modal content
  conditioning, pending larger and phase-restricted closed-loop utility tests.
- User confirmed idle-GPU evaluation, then explicitly requested an autonomous
  goal while asleep. No local training. Started matched50-env/500-step seeds
  300000/400000 main none/mean/zero and held-out-seed validation of checkpoints
  selected using old seed100000 (baseline200, no-fusion300, static-fusion300).
  seed300000 none/mean/zero scores .72/.70/.74; not evidence zero is better.
- Three seed400000 processes stopped advancing near16:12 UTC, main threads
  CPU-busy and GPUs idle. Kept logs/resource snapshot, ptrace was denied. After
  >5min no progress, terminated only owned evaluation/queue processes; preserved
  incomplete outputs as `_stalled_attempt1`, never scored. Retry adds90-second
  faulthandler stack dumps and1500-second timeout, disables video writing while
  keeping the checkpoint, simulator, seed,50 environments,500 steps unchanged.
  Original/zero retry on3/6; mean retry queued on6. GPU4 acquired another job
  temporarily; used it again only after idle snapshot. No other jobs terminated.
- Queued two-seed mean modulation restricted to pre-contact or post-contact,
  pre-success inputs, using only previously executed contact/reward history.
  Extended summarizer verifies selected histories are frozen and recomputes
  causal masks from physical trace. Added deterministic regression coverage.
  Context-slice intervention hooks are CPU-tested but not GPU-validated yet.
- Fixed hard per-head attention retains actual V/output projections. Relative
  fused-vision error main mean1.045e-5/max2.098e-4; Peg mean3.087e-4/max3.211e-3.
  Static affine approximation on cached features, not closed-loop equivalence.
- Full suite before matrix-verification test:119 passed,1 skipped,3 existing
  dataset-path failures. Matrix-specific suite including new test:10 passed.
- Stall follow-up found container memory.max=64GiB despite host251GiB; three
  evaluation processes retained~10.7GiB RSS each, including5GiB saved optimizer/
  model payload and unused training workspace. Cgroup memory-pressure full
  avg10~70%, checkpoint/image preprocessing repeatedly appeared in watchdog
  stacks, and millions of limit events were recorded. Not a confirmed PhysX hang.
  Evaluation now deletes payload and unused workspace after selecting/moving
  the saved EMA policy, then gc.collect(); no policy parameters, normalizer,
  scheduler, training code or simulator protocol changed. Preserved aborted
  memory-pressure attempts separately, retried original/zero seed400000 and
  no-fusion seed300000. After cleanup RSS~4.4GiB and pressure avg10~0.05%; direct
  before/after resource snapshots under results/logs. GPU4 externally occupied
  intermittently; each queued launch checks idle memory and waits at most600s.
- Final full suite with matrix verification:120 passed,1 skipped,3 existing
  missing-canonical-dataset-path failures. New Peg subset deliberately does not
  impersonate the absent complete training dataset. No training launched.
- Memory cleanup alone did not resolve three-way contention; corrected the
  initial diagnosis rather than treating the transient pressure drop as a fix.
  Reduced to at most two evaluations, advised DONTNEED only for clean pages of
  five checkpoints this investigation read (no file deletion/global cache drop),
  and retained all incomplete attempts without scores. Cgroup current dropped
  to~20GiB, full pressure decayed to0 and simulation resumed at~1.1steps/s.
  Original400000 retry now on3, zero400000 completed .62 on6, mean400000 running
  next on6. Remaining selected/fixed-phase runs stay capped at two concurrently;
  GPU4 queue waits for GPU6 static-fusion completion before starting. This is
  infrastructure pressure evidence, not a model-failure observation or a unique
  explanation of the kernel's reclamation behavior.
- Six full matrix evaluations completed and verified: seed300000 original/
  mean/zero=.72/.70/.74; seed400000=.60/.54/.62; pooled66/62/68 of100. All five
  initial-state arrays exact within seed,50x500 binary rewards agree with
  official mean_score, mean modulation constant and zero exactly zero. Paired
  mean-minus-original−4pp, conditional95% interval[−9,0], gained1/lost5,
  exact sign p=.21875; zero+2pp,[−3,7], gained4/lost2,p=.6875. No equivalence,
  superiority of zero, or training-seed replication claim. Report/plot under
  controlled_matrix_report; phase and preselected-checkpoint jobs still pending.
- Completed all16 valid full50-env/500-step runs across evaluation seeds300000/
  400000, with at most two concurrent after memory-pressure diagnosis. No new
  training/dataset download. The16 runs reuse100 initial conditions, not800
  independent starts. Between-seed initial-state max difference~.189, plug-pos
  ~.193; within-seed all five paired arrays exact. All original trajectories
  preserve saved EMA, normalizer and100-step sampler. Incomplete attempts excluded.
- Phase mean-only pre scores .70/.62 (pooled66); post/pre-success .74/.64
  (pooled69). Recomputed causal masks agree exactly; selected complete histories
  frozen. Relative to original66, pre0pp/95%[−7,7], gained6/lost6,p1; post+3pp/
  [−3,9], gained7/lost4,p.54883. Conditional environment intervals, not equivalence
  tests or training-seed variation. Do not add pre/post effects.
- Preselected baseline200 new scores .64/.44=54%; no-fusion300 .70/.58=64%;
  static-fusion300 .62/.56=59%; main100 .72/.60=66%. Selection fixed using old
  seed100000 before each new result, no reselection on new seeds. Main minus
  baseline+12pp, conditional95%[2,22], gained21/lost9, exploratory unadjusted
  paired sign p=.04277. Different saved epochs, one training seed; no isolated
  architecture or multiple-testing-adjusted significance claim. No-fusion64 is
  close to main66, so attention fusion necessity remains unproved. Old best-score
  gap2pp was a warning about selection, not proof that held-out improvement vanished.
- New paired geometry pooled baseline/main: success54/66, no-contact10/11,
  contacted-far27/15, contacted-near9/8 at2x success threshold. Of21 gained,
  15 were baseline contacted-far and6 contacted-near. Of9 lost,4 were main far,
  4 near,1 no-contact. This localizes descriptive geometric failure association,
  not causal time stage or scalar modality trust. Per-seed threshold sensitivity
  and pooled figure/JSON retained. Original nine-task benchmark table unchanged.
- Final static figures/PDFs: evidence_overview/mechanism_evidence and
  failure_geometry_pooled. Plot scopes and counts reviewed visually; captions
  moved off data marks. Reproduction scripts in result root, source analysis
  helpers in scripts/manifeel; final resource snapshot records own GPU exit.
  GPUs3/4/6 returned2MiB/0%; retained other jobs. Context-only GPU utility,
  closed-loop fixed-attention equivalence and extra fusion-HyperNet-specific
  causality remain untested, explicitly documented rather than implied complete.
