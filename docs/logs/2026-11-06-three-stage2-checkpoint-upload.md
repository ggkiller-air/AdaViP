# Selected completed Stage2 checkpoints to ModelScope

- User authorized 19 fixed checkpoints to `ggkiller/UVIP` via direct HTTPS,
  no proxy, no Git transport, no independent hash verification.
- Completed 2026-11-06 20:31:38 +0800: 64,569,207,267 bytes; all 19 remote
  filenames and byte sizes match. Pre-existing remote filenames/sizes unchanged.
- ManiFeel DP: E20/50/100/200/300/400/500/600 under `ManiFeel/stage2/`,
  preserving `epoch<N>.ckpt` names (8 files).
- Tabero DP: E20/50/100/150/200 under `Tabero/stage2/`, preserving
  `epoch<4-digit-N>.ckpt` names (5 files).
- Tabero-VTLA: steps5000/10000/20000/30000/40000/50000 under
  `Tabero/stage2/vtla/`, preserving `step_<7-digit-N>.pkl` names (6 files).
- Staged with hardlinks to immutable snapshots; four SDK workers, proxy
  environment variables removed before SDK import, remote sync/deletion off.
  Standard SDK protocol content identification unchanged; filename/size audit
  only. No latest files, training changes or extra assets uploaded.
- Manifest, uploader, log, remote-before listing and SUCCESS.json:
  `/data/wangzihao/outputs/modelscope_uploads/UVIP_three_stage2_selected_20261106/`.
