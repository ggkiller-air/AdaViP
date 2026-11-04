# L1 dropout10 and original L1 mask sensitivity

- User requested a fresh action-only L1 with10% input dropout, batch32,
  maximum100epochs and fixed E20/E40/E60/E80/E100 checkpoints. Then requested
  a separate complete original-L1 mask diagnostic before utility generation,
  with L0/L1/L1-dropout comparison to follow.
- Added a parameter-identical L1 mask encoder.90% samples are full;10% split
  uniformly among V1/V2/V3/T1/T2/P/L/vision/tactile. Post-projection sensor
  tokens including identity are zeroed, six flags reflect presence, late
  language is zeroed when absent. Both history frames share the condition.
  Independent dropout RNG is captured/restored with the original training state.
- Parent trainer adds overridable protocol/source/condition hooks while keeping
  its original full-input restriction by default. Step/epoch logs now expose
  condition counts; original and Future frozen production sources are untouched.
- Initial focused original-L1/Future/dropout regression:32passed in71.34s.
  Expanded checks verify identical original/dropout mask conditions, exact direct
  paired-loss agreement, noise independence from batch grouping and signed summary
  semantics. One expanded run passed15tests then hit the previously documented
  FileLock wait in the existing first-update/cache test; interrupted after183s.
  Isolated retry and full regression are running separately.
- Offline diagnostic defaults to original L1 E500 EMA, all400 demonstrations /
  71,265anchors, ten input conditions and four fixed paired noise draws. Saved
  arrays preserve every sample/repeat/condition loss, timestep and source index;
  signed deltas, per-task/global/macro and complete-window summaries are exported.
  This is training-data sensitivity, not held-out or simulator success.
- L0 native missing tokens differ from zero-token L1 masks; this is explicit in
  the comparison protocol. Use original L1 E100 for epoch-matched comparison with
  the new E100 dropout model; current E500 diagnostic answers the present-model
  question without attributing epoch differences to dropout.
- Current Slurm inspection: B407886 running on server13/H200143771MiB,
  GPU40977MiB/67%; trainer327198 uses17442MiB and existing separate memory holder
  336756 uses23516MiB. C/D/E407887--407889 and benchmark407932 pending on
  AssocGrpGRES. No existing job was modified. New tasks use isolated H200 jobs,
 32CPUs/384GiB, gpu,gpux/common queried maximum, locks and safe continuation.

## Final checks and submissions at17:04 CST

- Isolated existing cache/first-update retry passed in9.66s. Full repository
  regression then passed311tests,1skipped,5warnings in203.99s. Shell syntax,
  Python compilation, resolved launch configs and all478frozen source hashes
  passed. Complete test log is preserved with the experiment artifacts.
- Root: `/data/wangzihao/outputs/manifeel/l1_dropout10_and_mask_batch32_ep100_seed42_20261031_170151/`.
  `source/` is read-only; `source_sha256.json`, source revisions and per-run
  launch configs preserve the exact submitted working-tree implementation.
- Submitted diagnostic408795 (`l1_E500_mask_all4`) first, then training408796
  (`l1_drop10_batch32_ep100`). Each is one H200/32CPU/384GiB job, gpu,gpux,
 48h/common queried maximum (gpu100h,gpux48h), Requeue=1. Scheduler verified
  both PENDING/Priority; no diagnostic output or GPU preflight is claimed yet.
  No existing production job was changed. New training performs its real-data
  GPU preflight only after receiving its independent allocation.
- Outputs: `l1_epoch500_mask/` and `l1_dropout10/`; `submissions.json` and the
  queue JSON files record job IDs, commands and queried partition limits.

## Completed original-L1 mask diagnostic

-408795 completed on server52,19:46:21--20:04:33CST,18m12s,ExitCode0:0.
  The279GiB RAM cache load took427.97s; the full paired evaluation loop took
 590.40s. Single-timestep denoising scoring, no backward/update or100-step
  action sampling, and reuse of encoded observations across masks explain
  the short elapsed time. This is not a rollout or a new training run.
- Independently loaded losses.npy [71265,4,10], timesteps.npy and metadata.npy:
  all2,850,600loss values finite/nonnegative/nonzero; sample indices exactly
 0--71264,9tasks/400episodes, diffusion timesteps0--99, every four-repeat
  loss vector nonidentical. Recomputed float64 condition means exactly match
  report.json. Complete-window subset contains65,265anchors.
- Full loss0.00245374; masked/full loss ratios V1/V2/V3=15.00/31.49/46.36,
  T1/T2=13.05/16.18,P=87.84,L=42.20,vision=90.40,tactile=40.50.
  Complete-window ratios remain similar, so boundary padding does not explain
  the overall increase. This establishes strong OOD mask sensitivity, not a
  simulator success decrease or a calibrated causal sensor-value estimate.
-408796 remains PENDING/MaxGRESPerAccount; no dropout training result claimed.
  Audit details saved as l1_epoch500_mask/completion_audit.json.

## Comparison with the existing E500 K64 label archive

- Located the previous formal L0/Stage0 E500 EMA archive at
  `/public/home/wangzihao/projects/AdaViP/outputs/stage1_utility_labels_ep500_k64`.
  It contains65,265complete windows,64stratified/weighted draws,9signed
  utility labels plus per-window sampling variance. Teacher trained80%full /
 20%mask and uses native learned missing tokens; normalizer differs from L1.
- The new signed_deltas.npy is a [71265,9] K4 diagnostic utility estimate,
  not a regenerated K64 formal label archive. Matched all65,265old source
  windows against new metadata; task/episode/timestep/source index agree.
- Compared raw signed means on matched windows. New/old dimension mean ratios
  range11.18--68.83; per-window Pearson ranges-0.0601--0.1749. This is not a
  common rescaling of old labels. Full-input loss means are old0.00329359,
  new0.00257757, but normalizers/draw protocols differ, so not a controlled
  model-quality comparison. Larger deltas do not establish better labels.
- Saved all nine dimensions, exact joined protocols and caveats in
  l1_epoch500_mask/comparison_with_l0_e500_k64.{json,md}. No additional GPU
  work or label-generation job was submitted for this read-only comparison.

## Normalization comparison

- On the same65,265verified matched windows, independently normalized each
  model/dimension by its own RMS. Mean normalized utilities span old0.577--0.851
  versus new0.285--0.466. Cross-model per-dimension Pearson is unchanged under
  positive fixed scaling (-0.060--0.175); averaged vector cosine0.636.
- Also computed per-window signed u/sum(abs(u)) across nine interventions.
  Mean relative vectors cosine0.740, median0.783; largest-score dimension agrees
  on34.77%of windows. Mean P/L/vision/tactile weights old20.19/4.31/28.13/6.16%
  versus new24.20/9.93/23.07/10.91%. These overlap intervention groups, so they
  are relative scores, not additive causal sensor-contribution percentages.
- Per-dimension relative-weight Pearson/Spearman and both full normalized tables
  saved in normalized_comparison_with_l0_e500_k64.{json,md}. No model or original
  labels changed; these are descriptive scales, not new fitted training targets.
  K4/K64 noise, architecture, mask representation and normalizer confounds remain.

## Dropout preflight device failure and retry

-408796 failed on server13 at21:49:35CST after55s,ExitCode1:0, before any
  optimizer update or checkpoint. Traceback enters the first ResNet conv:
  CPUFloatTensor input versus CUDAFloatTensor weight. Root cause is the added
  preflight calling policy.to(cuda) before set_normalizer(); loading this custom
  normalizer reconstructs its ParameterDict on CPU and normalization moves the
  inputs back to CPU. This was a runner bug, not a data/dropout/learning-rate issue.
- Fixed runner order: load normalizer, move the complete policy to CUDA, then
  assert both all normalizer parameters and normalized observations are on the
  policy device. Preflight reports both devices explicitly. Added a CUDA-only
  regression covering the actual preflight, unchanged policy weights/optimizer,
  and restored CPU/CUDA/dropout RNG; it awaits a GPU allocation.
- Focused dropout/paired-diagnostic tests8passed/1CUDA-skipped in25.32s on the
  CPU node. Full repository regression started separately. No GPU validation
  result is claimed while the new independent allocation remains pending.
- Created read-only source_dropout_devicefix_20261031_223831,478hash-verified
  files. Only runner and regression test differ from the original frozen source.
  Existing training launch config, completed mask diagnostic and other jobs
  remain unchanged. No checkpoint exists, so restart preserves original seed42
  initialization rather than pretending to resume trained progress.
- Submitted one replacement409885, gpu,gpux/common queried48h, oneH200,
 32CPUs/384GiB, Requeue=1 and isolated per-output lock. Scheduler verifies
  PENDING/AssocGrpGRES. Original failed job and original source are preserved;
  submissions.json records the retry and training_history. Failure/retry
  provenance is dropout_devicefix.json at the experiment root.
- Prior to submission, inspected E407889's allocated H200143771MiB,41307MiB
  used/3%utilization, trainer1044495=17768MiB and existing holder1044560=23516MiB.
  B/C/D/E and runtime benchmark remain running; none was changed or canceled.

- Final fix verification: full repository311passed/2skipped/5warnings in232.25s;
  new CUDA regression is skipped on the CPU node. Test log preserved as
  devicefix_full_pytest.log. Final scheduler check still409885PENDING /
  AssocGrpGRES. Report the repair/resubmission separately from unperformed
  real GPU preflight; no successful formal update is claimed for this retry.
