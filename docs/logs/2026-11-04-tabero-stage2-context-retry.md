# Completed Tabero labels; fix Stage2 initialization check

- Job425476 FAILED1:0 at15:57:45+0800. All109097E150EMA K64 label windows
  had completed and finalized at15:57:41; no formal Student update or Student
  checkpoint existed. Failure was the pre-training CPU context comparison:
  exact activation equality rejected max absolute error1.1920928955078125e-7
  after the Teacher had been evaluated on GPU and moved back to CPU.
- Retain exact comparisons of every inherited encoder and normalizer tensor.
  Compare computed FP32 context activations with rtol1e-5/atol1e-6 and save
  the observed max error. No model weights, loss, masks, labels or normalizer
  definition changed. Regression allows1.2e-7activation roundoff but still
  rejects a1e-6inherited-parameter alteration.
-17dedicated/native tests pass in workspace and new immutable retry source.
  Full regression397passed,2skipped in219.07seconds.
  Existing source remains preserved. New source:
  `source_stage2_contextfix_20261104_161554` under the original Tabero run.
  Original label source_id remains recorded; launch config adds the separate
  Stage2 source identity. All finalized label arrays and shards are reused.
- Submitted retry426706 at16:17:49, normal yangyc/user_wangzihao,
  oneH200/24CPU/128GiB/gpu,gpux/48h/requeue and allocation-local16%VRAM buffer.
  It uses the same run directory and exact E150EMA/label identity; initial
  state PENDING. Registry now points to426706 and records425476 as previous.
- Progress refresher now reads bounded squeue/sacct metadata at its normal
  five-minute refresh. Failed jobs display failed; a submitted Student with
  complete labels displays queued until initialization/training begins.
  Already-started training still displays training during resource waits.
  Root document remains one table plus update time. CPU refresher423354 was
  requeued to load this change; no recovery/VRAM polling service was added.
  Its normal five-minute loop reloads the renderer for future reporting fixes.
-16:12live per-allocation nvidia-smi audit: all8running GPU jobs exceed15%.
  FTP six tasks each52159/143771MiB=36.279%; ManiFeel42081MiB=29.269%;
  Tabero-VTLA40531MiB=28.191%. UniVTAC DP was pending and425476 had exited,
  so neither held a GPU. Audit receipt:
  `/data/wangzihao/outputs/uvip_gpu_vram_audit_20261104_1612.json`.
