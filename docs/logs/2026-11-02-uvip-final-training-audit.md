# UVIP final training audit

- User fixed five experiment groups: DP+UVIP on ManiFeel/UniVTAC/OpenViTac,
  FTP-1+UVIP on UniVTAC/OpenViTac. Confirmed via explicit reply that FTP-1
  generates its own utility; DP uses L1 no-dropout. Stage2 follows S0 A/B/C.
- Rehashed seven aggregate files listed by ManiFeel L1 E500 K64 COMPLETE:
  sizes and SHA256 all match; 65,265 windows/255 shards. Did not rehash all
  shards or execute new GPU work.
- Reviewed native FTP source and public model lists; six UniVTAC task models
  exist, missing insert_HDMI/grasp_classify. Downloaded small insert_hole
  model/train/normalization configs only:120D/H32/adarms/bf16/no history,
  mix actions and non_tactile_dropout_ratio0.0. Other task configs, actual
  weights loading and remote OpenViTac checkpoints remain unverified.
- Identified L0-only Stage2 checkpoint/encoder assumptions, missing L1 adapters
  in UniVTAC/OpenViTac, OpenViTac MLP rather than S0 attention, FTP scalar loss,
  hardcoded training augmentation and state-adaRMS mask bypass.
- Detailed actionable plan: `docs/uvip_final_training_plan.md`. No training
  submitted, no production code edited. Existing uncommitted label-generation
  and test changes were left intact. Evidence cached under
  `/public/home/wangzihao/.cache/uvip-final-audit/`.

- User revision: all UniVTAC experiments use only the six FTP-covered tasks;
  FTP remains six separate single-task students with reused teachers. No
  additional FTP teacher training. Hyperparameters and phase budgets are
  model/data-specific; historical S0 numbers are references only.
- Rechecked official OpenViTac release links and HF API. No matched public
  OpenViTac advanced-policy checkpoint verified. NeoteAI task checkpoints
  have actual weight/norm files, but their model cards identify NeoSim at
  UniVTAC commit695a22d, rather than the OpenViTac benchmark. Updated plan
  with explicit release/compatibility distinction; no GPU work or downloads
  of full weights performed.

- Tabero replacement audit: downloaded both upstream source archives, with
  commits identified from archive comments. Field+FS49999 full params+norm
  (6.40GB), object25 dataset(11.29GB) and simulation USD/init assets are
  downloading under pinned revisions with per-file Git/LFS verification.
  Real completed parquet passed RGB decoding and finite shape checks.
- Verified native JAX flow-loss/TCN/force-position interface; generic PyTorch
  conversion lacks tactile coverage. No GPU or checkpoint forward executed.
- User asked for RTX rollout handoff: documented independent Sim5.0/Lab2.2
  environment, explicit paper versus minicase settings and bounded sensor/
  Teacher rollout gates. H200 handles model server;3090 remains untested.
  See docs/tabero_rtx_handoff.md and docs/tabero_uvip_feasibility.md.

- Deployment revision: user requires inference and simulation on the same RTX
  server without H200 connectivity, and explicitly allows separate3090 GPUs
  for model and simulator. Updated handoff with local weight/server setup,
  localhost connection, separate-GPU priority and same-GPU fallback, per-card
  VRAM/process telemetry and full firm/gentle episodes. No remote execution
  claimed. Source review found official libero_object runner hardcodes30chunks;
  corrected previous description that treated30as a shorter smoke horizon.

- Corrected final ManiFeel teacher selection after user clarification: L1
  no-dropout E50 EMA/global_step55700. Re-read all three manifests: historical
  L0 E500/547400 and separate L1 E500/557000 stores are both complete; final
  L1 E50 store is still generating under job416813. No final Stage2 job was
  submitted and no E500 final training started. Updated plan to prevent
  treating historical L1 E500 assets as final E50 labels.

- User confirmed FTP-1 retains utility prediction + four FiLM heads and native
  noisy-action queries; deleted proposed extra query/attention/action residual
  from both current architecture documents. Added tensor-level modulation and
  per-sample native masked flow-loss helper. Four dedicated CPU tests pass;
  native Teacher forward/integration is NOT yet verified. Six released model
  downloads continue independently (49.49GB total).
- UniVTAC final six-task L1 Stage1 implemented in UniVTAC repo. First attempt
  417347 failed before optimization because its frozen source omitted the parent
  policy/_base_policy.py. Preserved it; fixed dependency in a new snapshot and
  reran all checks from that isolated snapshot. Retry417405 on server23 H200
  passed the actual100-update gate:378.59samples/s,0.00648data wait. Raw/EMA/Adam
  checkpoint4,824,900,247bytes saved; independent archive hardlink retained under
  /data/wangzihao/archives/univtac/l1_six_task_20261102_233553/. No existing job stopped.
- Tabero Field+FS DP L1 implementation added: two independent ResNet18/GN
  histories, native-shaped bilateral marker TCN,7D state, late512D CLIP,13D DDPM.
  CPU cache417419 completed737episodes/109097frames/88prompts. Initial pending
  cache417415 was cancelled to replace its snapshot with the complete package.
  Real cache finite/shape/boundary/normalization preflight passes, stored under
  /data/wangzihao/outputs/uvip_final_implementation_20261102/.
  Ten dedicated tests passed in both workspace and isolated snapshot, including
  exact raw/EMA/Adam resume. Submitted Stage1 job417432:batch64,200epochs,341000
  updates,AdamW1e-4,warmup1000,FP32,force0.1,retained25epoch snapshots. Long
  training requires passing its first100actualupdates throughput gate.
- User provided https://github.com/ggkiller-air/Tabero as sync remote. Cloned
  /public/home/wangzihao/projects/Tabero, origin points there; ls-remote shows
  only main at f57f147c4abd0c223186878ff595e4a0089c0fbe, identical to our original
  archive. Asked for the pushed3090fix branch/commit; did not claim fixes synced.
  Reported successful RTX rollouts are user evidence, not local execution.
- Full pytest initially lacked diffusion_policy in PYTHONPATH; reran using
  source scripts/manifeel/common.sh + activate_manifeel (environment setup).
  Dedicated FTP/Tabero tests are already10/10 passing.

- Full repository suite with declared ManiFeel environment paths:366passed,2skipped,
  5warnings in181.20s. Both new Tabero sbatch scripts pass bash -n. Tabero417432
  allocated server26,physicalGPU0 UUIDGPU-582d0f8b-a458-2dbf-0bb8-df0cec8b5343,
  H200143771MiB initially0used/0util/no compute processes. Startup/preload in progress.
  Consolidated status: /data/wangzihao/outputs/uvip_final_implementation_20261102/implementation_status.json.

- Tabero417432 reached real optimization and passed its100-update gate: 632.86samples/s, data_wait=0.00877. First raw/EMA/Adam checkpoint saved and external archive hardlink retained.
  Recent GPU sampling:14333MiB,97%util; steady throughput about940samples/s.
  Six FTP task configs all verified120D/H32/adarms/gemma_small512D tactile;
  this config audit does not substitute for the pending full weight-load test.
