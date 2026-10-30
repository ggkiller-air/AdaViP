# Single-task DP Teachers: three tasks / eight runs

- User requested five image sensors, paired no-dropout/dropout training with
  full/removal60/40, batch8 and400epochs, then added two runs with the official
  wrist/right-TacRGB inputs and a state tokenizer in both. Selected Power Plug,
  USB and Ball Sorting for five-input pairs; the additional two-input pair is
  Power Plug. All use seed42 and preserve state, omitting language.
- Added independent encoder/policy/dataset/workspace and launcher; no existing
  trainer or third_party file changed. Five-input versions have six tokens;
  two-input versions have three. State Linear14->256/LN/SiLU and image-history
  projections feed Concat+MLP256. Camera/TacRGB inputs all256x256. Dropout is
  single-window token removal with learned missing tokens and presence flags:
  seven removal conditions for five inputs, two for two inputs, state retained.
- Retained official H16/To2/Ta8, native6D/7D actions, padded sequence windows,
  shuffled nonreplacement epochs, batch8 train/val, FP32, independent random
  ResNet18 with official GroupNorm, AdamW8e-5/betas.95/.999/wd1e-6,500-update
  warmup/cosine, EMA and100-step epsilon DDPM/U-Net[512,1024,2048]. No gradient
  clipping. Train-only normalizer differs intentionally from the pinned official
  loader's whole-store fit. Added token/fusion architecture, diagnostics and the
  requested400epochs are documented deviations; not a verbatim official model.
- Actual source checks: each task has50episodes, split49train/1val. Official
  train/validation sequence counts: Plug4932/143, USB5490/136, Sorting4748/128.
  Actual batches match all configured five image histories and state[8,2,7];
  action[8,16,6] for Plug/USB, [8,16,7] for Sorting.
- Nine focused regressions pass, covering all interventions, state/image
  gradients, masked-input invariance, native dimensions, official sequence
  parity, RAM parity, train-only normalization, matched initialization/noise,
  and exact mid-epoch raw/EMA/Adam/scheduler/RNG resume. Full suite:
  279passed/1skipped, five existing Transformer warnings,142.33seconds.
  Evidence: `/public/home/wangzihao/.cache/st/test/`.
- Read-only pre-submission GPU check via existing L1 allocation399733:
  H200143771MiB,29881MiB used,83% instantaneous utilization, PID1040006.
  No existing job was changed or canceled. New jobs request one H200,16CPUs,
  128GiB each; submit helper queries both partition limits (100h/48h) and
  submits exactly one `gpu,gpux` job per run at their common48h, requeue eligible.
  Per-run locks, explicit srun CPUs,40-update isolated preflight, CPU/worker
  affinity evidence, strict EMA/normalizer restore, atomic checkpoints and
  dependent safe-stop continuation are implemented.

## Submission at19:12CST

- Published independent branch `experiment/single_task_teacher_batch8_20261030`
  at `1efd9a7`; retained the pre-existing dirty six-trainer status log untouched.
  Frozen2170 tracked main/submodule files with SHA256 inventory under
  `/data/wangzihao/outputs/manifeel/single_task_teachers_batch8_ep400_seed42_20261030_191203/source`.
  Source files are read-only; CPU evidence and source manifest are siblings.
- Submitted exactly eight jobs, each single `gpu,gpux`,48h,Requeue=1:
  Plug five nodrop404581/drop404582; Plug two nodrop404583/drop404584;
  USB five nodrop404585/drop404586; Sorting five nodrop404587/drop404588.
  Every per-run queue JSON records current partition MaxTime100h/48h and the
  chosen48h. Root `submissions.json` maps all task/input/dropout/job/output pairs.
- At19:13 all eight are PENDING/AssocGrpGRES, with no allocation, stdout,
  preflight result or formal update yet. Current jobs399040/399733/399747 remain
  RUNNING and untouched. Do not claim GPU preflight or training has begun;
  allocation will automatically run isolated preflight then formal training.
- Latest focused rerun, including strict inference-only workspace restore and
  numerical EMA action equivalence:9passed in16.97seconds. Shell syntax and
  diff checks pass; dry-run submission produces exactly eight entries.

## Complete controls and runtime request

- Added USB two-input jobs404624/404625. For paper-view consistency, Sorting
  uses front + right TacRGB, whereas Plug/USB use wrist + right TacRGB. Canceled
  only the just-submitted, still-pending wrong-view Sorting404626/404627;
  corrected front-camera jobs are404632/404633. The original eight jobs were
  not modified. All twelve retain state tokenizer, batch8/400epochs/seed42.
- Config/front-view support is committed at652386c. Corrected Sorting uses
  immutable `additional_source/`, hard-linked from the original snapshot except
  for the five replaced source/doc/test files. SHA256 inventory is in
  `additional_source_manifest.json`; `all_twelve_submissions.json` lists exactly
  twelve active experiment jobs, preserving canceled submissions separately.
  Ten regressions pass, including two-input7D and task-specific camera checks;
  actual USB/Sorting batches match[8,16,6]/[8,16,7] and state[8,2,7].
- Plug404581--404584 passed40-update H200 preflight, strict EMA/normalizer
  restore, actual16-CPU trainer/worker affinity and first-step gradients, then
  began formal training. Observed steady throughput before runtime tests:
  five-input approximately44--45samples/s, two-input63--65samples/s. RAM decode
  completed before training; DataLoader wait approximately0.2--0.3%.
- User requested throughput checks without increasing batch and explicitly
  authorized inert VRAM occupancy per administrator's15% requirement. Current
  two-input H200 readback was7607/143771MiB,42% instantaneous utilization.
  CPU supervisor404638 adds a per-run holder within each allocated teacher job,
  reserving24017601822bytes (16% device capacity) in an independent zero tensor.
  No model/optimizer/checkpoint/RNG is changed. Existing four holders verified;
  an actual two-input GPU now uses31129MiB (21.65%). Supervisor follows queued
  runs and their continuation queue IDs; unrelated jobs are excluded.
- Independent convolution benchmark temporarily SIGSTOPs only Plug five-input
  nodrop trainer2722163, benchmarks its allocation at batch8 with unchanged
  FP32/TF32 flags and identical checkpoint/batch, then SIGCONTs it in finally.
  Tests compare baseline, cuDNN autotuning and channels-last image backbones;
  reservation remains active. No production convolution setting changed yet.

## Verified runtime outcome

- Three30-step timing windows per profile at batch8: baseline
  [46.221,46.373,46.296]samples/s; cuDNN autotune
  [46.346,46.058,46.859]; channels-last+autotune
  [46.499,46.577,45.987]. Means46.297/46.421/46.354, gains0.27%/0.12%,
  within timing variation. Retained production convolution settings. Matmul
  TF32 stays false and cuDNN TF32 true throughout; first-loss layout difference
  was approximately3.7e-6. Benchmark held identical batch/checkpoint/optimizer
  settings and did not increase batch or introduce AMP.
- `convolution_benchmark/results.json` and `trainer_resumed.json` preserve
  evidence. Target trainer was paused100.81seconds, then SIGCONT confirmed;
  no optimizer/RNG/checkpoint state was changed by the independent benchmark.
- VRAM holders are separate Slurm steps in the corresponding existing GPU
  allocations, supervised by CPU-only404638 (one CPU/1GiB, maximum5days).
  Each uses an exclusive per-run lock, no RNG, no recurrent GPU computation;
  GPU job termination releases its tensor. Actual active step listing confirms
  trainer and holder share the intended allocated GPU only. Other trainers
  continued advancing at approximately65samples/s after reservation.
