# Full-sensor Stage-0 foundation (2026-10-17)

- User approved a fresh 3-camera + bilateral TacRGB + proprio + language
  multi-task DP, configurable 80% full / 20% nine-removal dropout, train-only
  normalization, episode-local windows, time/dimension loss masks and signed
  per-sample paired denoising interfaces. Only new Stage-0 paths added; retained
  existing user edits/experiments and did not modify `third_party/`.
- Added schema, dataset, seven-token encoder/simple fusion, reference DDPM,
  independent EMA-aware workspace, batch8 config, CLI, source inspection and
  bounded Slurm preflight. Checkpoints preserve raw/EMA/config/normalizer/
  optimizer/RNG/EMA step and reject changed split/statistics/protocol on resume.
- Focused CPU regression: 28 passed in 16.90s, using project Python
  `/public/home/wangzihao/.local/miniforge3/envs/manifeel/bin/python`,
  `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`,
  `PYTHONPATH=$PWD:$PWD/third_party/diffusion_policy`,
  `-m pytest -q tests/test_full_sensor_stage0.py
  --basetemp=/data/wangzihao/pytest-tmp/full_sensor_stage0_20261017_v3`.
  Includes seven-branch gradients, all nine no-leak masks, epsilon/sample
  paired invariants, padded-slot gradient exclusion and bitwise CPU epoch
  resume/checkpoint loss equality. Subsequent small hardening rechecked below.
- Real source inspection read all nine stores and two boundary windows per
  task. Default split: 391/9 train/validation episodes, 69,205/2,060 frames;
  finite train-only normalizer. Output:
  `/data/wangzihao/outputs/manifeel/full_sensor_stage0_audit_20261017/dataset_check.json`.
  Reproduce with `scripts/manifeel/check_full_sensor_dataset.py --output ...`
  in the project environment. Source frequency remains unknown, not assumed10Hz.
- Whole repository CPU suite: 162 passed, 1 skipped, 1 failed (39.53s).
  Existing `test_manifeel_eval_config.py::test_film_checkpoint_uses_local_assets`
  cannot locate required `third_party/CLIP` source. No unrelated code/assets
  changed or downloaded. Command: same project Python/thread limits, add
  `$PWD/third_party/manifeel:$PWD/third_party/manifeel-isaacgymenvs` to
  `PYTHONPATH`, `-m pytest -q
  --basetemp=/data/wangzihao/pytest-tmp/full_sensor_stage0_all_20261017`.
  Full output:
  `/data/wangzihao/outputs/manifeel/slurm/full_sensor_stage0_pytest_all_20261017.log`.
- GPU request authorized by user. At submission, server37 exposed no GPU;
  existing interactive job366142 left untouched. Submitted job367844 at
  12:24:02+08:00 via `sbatch slurm/manifeel/preflight_full_sensor_stage0.sbatch`,
  requesting one H200-partition GPU,8 CPUs,64GiB,30min. Two batch8 train steps,
  one full+nine-mask validation batch, checkpoint load; no formal training.
  Initial state PENDING/Priority. Job logs:
  `/data/wangzihao/outputs/manifeel/slurm/full_sensor_stage0_batch8_preflight_367844.{out,err}`.
  This is not GPU validation success until a completed successful result exists.
- Job367844 completed successfully (exit0,1min49s,server51). Initial allocation
  reported NVIDIA H200,143771MiB total,0MiB used,0% utilization,no compute jobs.
  Native image shapes, full-width U-Net and EMA passed two optimizer steps
  with finite loss/gradients, all ten paired validation conditions, atomic
  latest/best/epoch1 saves, then a fresh process restored epoch1/global_step2
  and identical manifest/normalizer/protocol fingerprints. Peak allocated
  CUDA memory8.134GiB; measured first-two-step throughput2.75samples/s includes
  startup and is not a steady-state performance estimate. Loss1.19461,
  full validation1.18383; these untrained smoke losses are not performance.
  All16 stochastic training draws happened to be full input; no claim of
  GPU training coverage for all nine removals. Paired GPU eval covered all
  nine; explicit forced-mask backward coverage is in CPU regressions.
  Bounded validation covered peg insertion only (coverage1/9); source checks
  covered nine tasks. No rollout or trained-teacher gate claimed.
- Full implementation/API/limits: `docs/full_sensor_stage0.md`. No simulation
  rollout, trained-teacher selection, utility cache, Stage1/2 or long training.
- Final whole-suite recheck after hardening/extra forced-mask, singleton-stat
  and non-EMA validation tests:165 passed,1 skipped, the same unrelated FiLM
  source-asset test failed (38.09s). Output:
  `/data/wangzihao/outputs/manifeel/slurm/full_sensor_stage0_pytest_all_final_20261017.log`;
  basetemp `/data/wangzihao/pytest-tmp/full_sensor_stage0_all_final_20261017`.
  GPU smoke artifacts/checkpoints total approximately14GiB, retained outside Git.
- Final focused regression:31 passed in19.49s, including explicit nine-mask
  backward and both EMA/raw validation paths. Same project Python/thread
  limits, `-m pytest -q tests/test_full_sensor_stage0.py
  --basetemp=/data/wangzihao/pytest-tmp/full_sensor_stage0_focused_final_20261017`.
  Output `/data/wangzihao/outputs/manifeel/slurm/full_sensor_stage0_pytest_focused_final_20261017.log`.
  Shell syntax and whitespace checks passed; `third_party/` has no changes.
