# Direct training-checkpoint release

- User authorized direct HTTPS API uploads to `ggkiller/Manifeel-ckpts1018`,
  with proxies removed, no Git, and no hash verification.
- S1 requested epochs100/200 go under
  `stage2_encoder_ablation_t2_batch32_seed42_20261028/s1/checkpoints/`.
  L1 checkpoints go under
  `language_conditioning_t2_batch64_seed42_20261028/l1/checkpoints/`.
  Existing remote names/byte sizes are used to skip files already present.
- Single-task remote prefix chosen:
  `single_task_teachers_batch8_ep400_seed42_20261030/`, containing all12
  experiment names and their checkpoints (fixed epochs,best,final latest).
  A still-running experiment's latest is deferred until its completion event.
- Persistent hard-link staging, size-only manifest, process lock, per-group
  completion markers and API log are under
  `/data/wangzihao/outputs/manifeel/modelscope_uploads/training_release_20261031/`.
  The uploader removes every environment variable ending `_proxy` before SDK
  import, uses four API upload workers and resumable upload cache, and never
  deletes remote files. No independent hash calculation/comparison is used;
  the SDK performs content identification required by its upload protocol.
- Started11:07CST. Completion is established only by `SUCCESS.json` after
  remote filenames and byte sizes match every entry in the manifest.

- At11:09:38 S1 completed (E100 already present; E200 newly uploaded),
  with remote names/byte sizes matched. L1 completed11:12CST (four existing
  fixed snapshots plus newly uploaded E400,E500,final latest), with seven
  remote names/sizes matched. Single-task transfer then began; full-suite
  completion is pending. Uploader PID1785334 is active in the persistent
  command session; SDK retries/resumable cache and per-group locks are active.

## Reduced single-task release at12:36--12:39CST

- User requested stopping the original uploader, retaining only best.ckpt
  and epoch100/200/300/400.ckpt per single-task run, and deleting other
  already-uploaded single-task snapshots where possible. S1/L1 scope remains
  as previously requested. Verified original uploader command then terminated
  only PID1785334; no trainer or unrelated upload was stopped.
- Nine single-task groups had90checkpoint files committed. Selected scope
  is60files,45already present,15missing across the remaining three groups.
  All12trainers have now produced their epoch400 checkpoint, including Sorting
  five/drop. New upload_selected.py/selected_manifest.json use the original
  shared lock and cached upload state, direct HTTPS without proxy environment
  variables or Git, checking names/byte sizes only.
- Attempted deletion of45precisely enumerated extra files (E50/150/250/350
  and latest) under the single-task prefix. Server rejected the first delete
  with401/code10010101003: API token cannot delete; web console required.
  Zero remote files were deleted. deletion_manifest.json and
  DELETION_BLOCKED.json preserve scope and restriction; no bypass attempted.
- Replacement uploader started in a persistent command session, logs in
  upload_selected.log, completion only when SELECTED_SUCCESS.json exists.

- Replacement upload completed12:50:00CST, exit0. All60selected checkpoints
  (283714742976bytes) are present with matching names/byte sizes; only15
  missing files were committed, with zero upload failures. Final independent
  metadata listing confirms60selected plus45extra checkpoints remaining
  because API deletion is restricted. No further upload process is running.
  SELECTED_SUCCESS.json and FINAL_SELECTED_AUDIT.json preserve the result;
  DELETE_IN_WEB.md links the nine directories containing the45extra files.
