# CPU-only HyperResNet demonstration probes (2026-10-10)

- User authorized work that can proceed without idle GPUs. Excluded the three
  camera-mismatched tasks (`peg_reorientation`, `object_search`, `ball_sorting`).
  No `attention_trace.npz` was found in the existing evaluation output tree.
  The standard local dataset root contains only `plug_quan_Aug02` (and its archive),
  so other correct-camera tasks were not probed or downloaded.
- Added non-intervening encoder hooks and a bounded CPU demonstration probe.
  Uses saved EMA encoder configuration/weights and observation normalizers,
  upstream-compatible RGB resizing, and evaluation-mode transforms. Does not
  instantiate the simulator or action backbone. Regression tests verify exact
  encoder-output preservation, residual calculations, optional fusion HyperNet,
  episode-boundary sampling, saved normalization, and actual figure generation.
- Passed an eight-frame, batch-8 CPU preflight, then processed episodes 0/1/2,
  64 evenly spaced frames each, batch 8, two Torch threads, no loader workers,
  with `CUDA_VISIBLE_DEVICES=''`. Ran selected primary epoch 100 plus extra
  fusion HyperNet epochs 100/300 on the same 192 frames each. Outputs and
  separate process logs are under
  `/data/wangzihao/outputs/manifeel/probes/hypernet_batch8_20261010/`.
  Reproduction commands and interpretation are in `docs/manifeel_hypernet_probes.md`.
- Primary checkpoint is the canonical HyperResNet fusion epoch 100, not FiLM
  or the extra fusion HyperNet model. Its total feature-relative modulation
  means are vision 3.05% / tactile 28.98%; deviation from episode-mean modulation
  means are 1.16% / 10.47%. Raw feature norms differ strongly (11.06 / 1.09),
  so relative percentages must not be treated as cross-modal contribution shares.
  Sampled attention is almost fixed: three heads read TacRGB and one vision.
  These descriptive demo observations do not establish phase-dependent policy
  reliance or explain benchmark gains causally. No new success rate was measured.
- All eight RTX 3090s still had existing compute processes at the final check;
  GPUs 1/2/4/6 were actively utilized, and the others retained substantial memory.
  No GPU job was launched, stopped, or replaced with an irrelevant CPU simulator check.
- Validation after `source scripts/manifeel/common.sh; activate_manifeel`, with
  CUDA hidden and two CPU threads: `python -m pytest -q
  tests/test_manifeel_hypernet_probe.py tests/test_manifeel_attention_trace.py`
  passed (11 tests). Full suite: 90 passed, 1 skipped, 3 failed because existing
  profile/launcher tests require absent `pih_quan_June06` and `sorting_quan_Aug8`
  datasets. Details are in `pytest_full_final.log`; no unrelated launcher or
  dataset checks were changed.
