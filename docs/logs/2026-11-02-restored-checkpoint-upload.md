# Restored batch64/E500 fixed checkpoint upload

- User authorized uploading only L1 dropout E50/E100 and currently existing
  prescribed fixed snapshots for Future B/C/D/E to ggkiller/Manifeel-ckpts1018.
  Explicit constraints: direct connection, no proxy, no Git, no hash validation;
  exclude latest.ckpt. This authorization does not cover later snapshots.
- At13:44CST froze a12-file plan /61,021,758,990bytes from
  /data/wangzihao/outputs/manifeel/l1_restored_batch64_ep500_seed42_20261101/.
  L1 dropout E50/100; B,C E50/100/200; D,E E50/100. Staging uses hardlinks to
  these fixed snapshots; all source/stage sizes are checked without reading
  payloads for independent checksums. SDK performs protocol content handling.
- L1 destination:
  language_conditioning_t2_batch64_seed42_20261028/l1_dropout10_t2_batch64_ep500_seed42_20261101/checkpoints/
- Future root: stage1_future_prediction_k4_t2_batch64_ep500_seed42_20261101/,
  with B_future_t_raw, C_future_t_latent, D_future_vt_raw, E_future_vt_latent
  each containing checkpoints/. Initial authenticated API listing confirmed
  both destination prefixes absent. No pre-existing remote file replaced.
- Uploader removes all *_proxy variables before importing modelscope_hub;
  direct HTTPS API only, no Git commands. Per-uploader process lock and frozen
  manifest avoid duplicate launches. Filename/byte-size checks only.
- Persistent uploader, manifest, hardlink staging and log:
  /data/wangzihao/outputs/manifeel/modelscope_uploads/l1_restored_fixed_20261102/
- Completed at13:59CST: all12files /61,021,758,990bytes uploaded successfully;
  every group succeeded on its first transfer attempt. L1 completed13:47:28,
  B13:50:23, C13:53:37, D13:56:27, E13:59:05. Zero files deleted.
- Initial final-list assertion incorrectly treated API tree entries as files.
  Filtered FileInfo.is_dir, then reran filename/size auditing; existing matching
  files were skipped without retransmission. Successful exit0, SUCCESS.json and
  FINAL_AUDIT.json confirm exactly12requested paths and matching byte sizes,
  with no latest or other unrequested files under either new destination prefix.
- No proxy environment variables or Git commands were used. No independent
  hash calculation or comparison was performed; remote verification used only
  file names and sizes. SDK protocol-required content handling was unchanged.

## E200 additions at 17:37 CST

- User authorized adding exactly the two newly reported fixed snapshots:
  L1_dropout10/checkpoints/epoch200.ckpt (5,009,137,127bytes) and
  D/checkpoints/epoch200.ckpt (5,046,834,982bytes) to their existing remote
  directories. No later snapshot or latest is included.
- Direct HTTPS SDK transfer completed17:37:40CST, exit0, two files committed
  on the first attempt,10,055,972,109bytes total,136.2s SDK elapsed. No files
  deleted. No proxy variables, Git commands or independent hash verification.
- Final API listing matches both requested filenames and byte sizes; prior
  files under both destination checkpoint folders retain their names/sizes.
- Manifest, upload log, SUCCESS.json and FINAL_AUDIT.json:
  /data/wangzihao/outputs/manifeel/modelscope_uploads/l1_dropout_D_epoch200_20261102/
