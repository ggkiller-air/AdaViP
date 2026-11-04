# Current L1 DP assets to ModelScope

- User authorized direct API uploads to `ggkiller/UVIP`, benchmark folders
  `ManiFeel/`, `UniVTAC/`, `Tabero/`, each with `stage1/`, `utility_label/`,
  and `stage2/`. Explicit follow-up excludes all `latest` checkpoints.
  No proxy, Git transport, or independent hash verification.
- Frozen manifest at2026-11-03 19:25:55+0800:573files/76,676,611,784bytes.
  Sixteen fixed checkpoints: ManiFeel L1 E50 only; Tabero E25 throughE200
  every25epochs; UniVTAC E50 throughE350 every50epochs. Immutable payloads
  staged with hardlinks; mutable logs copied to point-in-time snapshots.
- ManiFeel includes all522original E50 utility files (65,265windows,
  255shards), configurations and provenance. DP Stage2 is unstarted for all
  three current L1 experiments. UniVTAC/Tabero labels are pending selection;
  directory READMEs state these facts rather than including historical or
  advanced-method weights. Stage1 loss charts/CSVs accompany Tabero/UniVTAC.
- Persistent uploader, frozen manifest/staging and log:
  `/data/wangzihao/outputs/modelscope_uploads/UVIP_dp_20261103/`.
  All proxy environment variables removed before SDK import. Four HTTPS
  upload workers; resumable SDK cache; filename/byte-size audit only.
  SDK protocol-required content identification is unchanged. No remote
  deletion, source mutation, or training changes. Completion requires
  `SUCCESS.json` and `FINAL_AUDIT.json`.

- Completed2026-11-03 19:40:41+0800, exit0. All573planned paths match remote
  byte sizes;16fixed checkpoints/76.68GB total assets, zero latest files.
  ManiFeel E50 completed19:26:50, labels19:27:56, Tabero19:34:01,
  UniVTAC19:39:57. No transfer failures or remote deletions.
- Initial final audit expected every pre-existing root file size unchanged;
  ModelScope automatically extended `.gitattributes`2075->2212bytes with
  LFS routing for the two production `progress.jsonl` files. Confirmed via
  direct small-file HTTP read. README/configuration remain unchanged, and
  every asset already matched. Adjusted that audit to record this specific
  platform metadata change, reran metadata-only checks, and wrote SUCCESS
  and FINAL_AUDIT without retransferring assets or altering .gitattributes.
