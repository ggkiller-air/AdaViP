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
