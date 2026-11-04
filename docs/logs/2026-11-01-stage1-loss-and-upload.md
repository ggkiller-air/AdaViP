# Completed Stage1 loss figures and direct checkpoint release

- Verified B407886/C407887/D407888/E407889 and L1-dropout409885 all completed
  E100/222800updates with ExitCode0:0. Each metrics.jsonl contains exactly
  epochs1--100. Fixed E20/E40/E60/E80/E100 checkpoints exist for all five.
- Rendered two220dpi PNGs with unsmoothed complete-epoch means under
  `/data/wangzihao/outputs/manifeel/analysis/stage1_future_l1_loss_20261101/`:
  `BCDE_loss.png` (Action/Total plus separately plotted raw/latent Future Loss,
  with late-action zoom), `L1_dropout10_loss.png` (overall Action Loss and
  sample-count-weighted full/missing-input losses). Plot script, CSV and JSON
  source/end-point records are saved alongside the PNGs. Visually inspected.
- Final Action Loss B/C/D/E:.01246286/.01216929/.01221757/.01205265.
  L1-dropout overall/full/missing:.01404156/.01338773/.01994212. These are
  training losses, not rollout scores or a controlled comparison at validation.
- User additionally authorized direct HTTPS upload of exactly25fixed snapshots
  to `ggkiller/Manifeel-ckpts1018`, no proxy and no hash verification. Independent
  API uploader/staging/size manifest/logs reside in
  `/data/wangzihao/outputs/manifeel/modelscope_uploads/stage1_future_dropout_ep100_20261101/`.
  All *_proxy variables were removed before SDK import. No independent content
  hash calculation/comparison is performed; the SDK handles content identification
  required by its upload protocol. Remote checks use filenames and byte sizes.
- L1 destination:
  `language_conditioning_t2_batch64_seed42_20261028/l1_dropout10_t2_batch32_ep100_seed42_20261031/checkpoints/`.
  Four future groups use new root
  `stage1_future_prediction_k4_t2_batch32_ep100_seed42_20261031/`, with subfolders
  `B_future_t_raw`, `C_future_t_latent`, `D_future_vt_raw`, `E_future_vt_latent`,
  each containing `checkpoints/epoch{20,40,60,80,100}.ckpt`.
- Transfer started11:33CST.25files total127223674759bytes. No existing remote
  path is overwritten; new-folder collision detection and per-uploader lock are
  active. Completion is claimed only after SUCCESS.json and remote size listing.

## Completed release

- Uploader completed11:59:37CST, exit0,25/25checkpoints committed. Per-group
  completion: L1 11:39:14, B11:44:12, C11:49:23, D11:54:23, E11:59:37.
  All groups succeeded on their first attempt; no existing files were replaced.
- SUCCESS.json records all25files/127223674759bytes and exact remote prefixes.
  Independent final API listing matched all25remote filenames and byte sizes,
  with proxy variables absent and no hash verification. FINAL_AUDIT.json
  preserves the name/size listing. No checkpoint downloads were needed.
- PNGs are3300x2200 (B--E) and3300x1210 (L1 dropout10); both were opened for
  visual inspection. Final inset axis uses four-decimal ticks to distinguish
  the late-epoch Action Loss curves. Plot data and script remain beside PNGs.
