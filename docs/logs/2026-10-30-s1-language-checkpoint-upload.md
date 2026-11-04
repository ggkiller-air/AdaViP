# Direct S1 and language-checkpoint upload

- User requested S1 epochs20/50/100 under existing ModelScope prefix
  `ggkiller/Manifeel-ckpts1018/stage2_encoder_ablation_t2_batch32_seed42_20261028/`,
  plus L1/L2 epochs50/100/200/300 in a new folder. Explicit requirements:
  direct connection, proxies unset, no Git, no hash verification.
- At18:46CST, ten requested files exist (45.45GiB); L2 epoch300 does not
  exist yet. L1/L2 remote prefix chosen:
  `language_conditioning_t2_batch64_seed42_20261028/`.
- Persistent API uploader PID367747 launched18:47CST; all `*_proxy`
  environment variables removed before SDK import. Only filename/byte-size
  checks are performed after upload; no independent SHA256 calculation or
  comparison. The SDK still performs its required LFS content identification.
  No Git commands or transport are used for this task.
- Hard-linked staging, size-only manifest, script and logs:
  `/data/wangzihao/outputs/manifeel/modelscope_uploads/s1_l12_20261030/`.
  Separate S1/initial-language completion markers prevent duplicate groups
  on restart. Once the initial ten finish, the same locked process watches
  for L2 epoch300 every300seconds (up to72hours) and uploads it automatically.
  No training process, checkpoint content or configuration is changed.
- S1 upload completed18:50:50 (3committed,0failed,229.0s). Initial LC
  upload completed18:58:06 (7committed,0failed,434.9s). At18:58:07 the
  remote names and byte sizes matched all10local entries; no hash comparison.
  `S1_SUCCESS.json`, `LANGUAGE_INITIAL_SUCCESS.json`, `INITIAL_SUCCESS.json`
  preserve results. Process367747 remains waiting every300seconds for
  missing L2 epoch300; `DEFERRED_PENDING.json` records its source/destination
  and deadline. All11files are complete only when `SUCCESS.json` appears.
