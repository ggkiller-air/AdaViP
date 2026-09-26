# 2026-09-26 Five-Job Loss Comparison and Checkpoint Upload

- Plotted jobs `320004`, `320005`, `320008`, `320009`, and `320014` through
  epoch 300 in `loss_320004_320005_320008_320009_320014_epoch300.png`.
  Training curves are per-epoch means; validation curves use the logged
  validation loss at each evaluation epoch. Both panels use logarithmic loss
  axes. Epoch-300 validation losses are 0.379957, 0.127554, 0.432592,
  0.434535, and 0.417368, respectively. Job `320005` trains Ball Sorting;
  the other four train Power Plug, so its loss is not directly comparable.
- Verified all 15 epoch-100/200/300 workspace checkpoints had readable
  PyTorch ZIP metadata, then ran
  `bash scripts/manifeel/upload_0926_ablation_checkpoints_modelscope.sh`
  with proxy variables unset. Uploaded to `ggkiller/multi-fm` on `master`
  under `0926/<run_name>/checkpoints/latest_epoch<epoch>.ckpt`.
- ModelScope's remote file API reported exactly 15 checkpoint paths under
  `0926`, and every remote size matched its local file. Total size:
  83,392,289,982 bytes. Upload log:
  `/data/wangzihao/outputs/manifeel/upload_0926_ablation_checkpoints_modelscope.log`.
  The ManiFeel environment test suite passed: 61 passed, 1 skipped.
