# Restored Future and dropout training loss plots

- November 2 13:39CST: plotted current November 1 batch64/E500/seed42 runs,
  rather than historical batch32/E100 runs. All five jobs remain RUNNING.
- Snapshotted metrics.jsonl and verified contiguous complete epochs and
  global_step=1114*epoch; checked plotted losses are finite and nonnegative.
  BCDE includes action, total, raw future and latent future panels, with
  E100+ action detail. Dropout includes overall action, full-input and
  sample-count-weighted missing-input curves, with E100+ overall detail.
  No smoothing or partial-epoch points. Raw/latent objectives use separate axes.
- Epoch/action/future at capture: B226/.0080657263/.00011579518;
  C204/.0087292364/.011257337; D174/.010249975/.00064368733;
  E137/.012157750/.013910182. Dropout186 action=.010519364,
  full=.0099572920, missing=.015614124. Weighted condition losses match the
  overall logged dropout loss for every epoch to rtol1e-9.
- Formal checkpoints inventory: B,C retain epoch50/100/200 plus latest;
  D,E,dropout retain epoch50/100 plus latest. Total17 files /86,466,500,649
  bytes at capture (rolling latest can change). Verified nonempty sizes;
  checkpoint contents were not deserialized. No training state modified.
- PNG/PDF pairs, CSVs, frozen metric copies, plotting script and summary.json:
  /data/wangzihao/outputs/manifeel/analysis/l1_restored_losses_20261102_1338/
- PNGs visually inspected. Future latent panel has explicit numeric log ticks.
