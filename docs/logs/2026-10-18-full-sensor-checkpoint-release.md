# Stage0 checkpoint comparison release

- User requested pushing Stage0 code and publishing latest plus epoch18 to
  `ggkiller/Manifeel-ckpts1018`, direct connection with proxies unset and no
  Git transport for ModelScope. Left training370163 and interactive366142 intact.
- Frozen at2026-10-18T20:38:35+08:00 under
  `/data/wangzihao/outputs/manifeel/checkpoint_archive/full_sensor_20261018_2040`.
  Checkpoint files are copied before upload; atomic training replacements do
  not change these snapshots. Latest:276completedepochs,step299000,cursor368;
  epoch18:step19476,cursor0, copied frombest. Both contain raw/EMA,optimizer,
  saved normalizer and workspace state, not deployment-only weights.
- ModelScope CLI folder API completed6files/0failures in171.8seconds;
  remote prefix `full_sensor_stage0_batch64_20261018/` includes both checkpoint
  files, task_embeddings.npz,training_config.yaml,manifest.json,README.md.
  Independently checked remote byte sizes AND SHA256 against local manifest:
  latest5008128886bytes,
  `daf98e462c8ff8cec48993c37f78d5575a11cc5a43703626759df305bcd4686d`;
  epoch185008126966bytes,
  `49a268def03a064215c183947b68de25a3cf4facff4dc2243676c79875f4f67d`.
  Upload log: `/data/wangzihao/outputs/manifeel/slurm/upload_full_sensor_modelscope_20261018.log`.
- Added separate Stage0 evaluation loader/runner entrypoints. Restore EMA by
  default, including saved normalizer, without optimizer/dataset allocation;
  enforce all sensor shapes, inject[B,To,512]language, truncate6D/7D action,
  preserve320x240TacRGB. Child importsIsaacGym before project imports/torch;
  tasks run in isolated subprocesses. Aggregate officialtest/mean_score rather
  than old info-window success_rate. Checkpoint comparison shares policy and
  simulator seeds. No third_party or legacy evaluation/training code edited.
- CPU restoration of actual frozenepoch18 checkpoint succeeded with strict
  loading, all8normalizer entries,eval mode and2-frame history. No GPU simulator
  rollout was run in this release: first verify with2env smoke on the evaluation
  machine, then50env/task/500steps. Do not infer simulator success from loss.
- Broad Stage0 tests79passed in99.77seconds; after adding aggregation/import-order
  guards, focused evaluation/plot suite9passed in6.49seconds. Full repository:
  215passed,1skipped,1known unrelatedFiLM local-CLIP-asset failure in116.29seconds.
  Logs `full_sensor_release_tests_20261018.log` and
  `full_sensor_release_repo_tests_20261018.log` in the persistent Slurm log root.
- Git remote has onlymain, identical to localHEAD before this release. Publish
  only Stage0 files and selected logs to existingmain without force; leave
  unrelated HyperResNet edits and PDF/PNG deletions uncommitted.
