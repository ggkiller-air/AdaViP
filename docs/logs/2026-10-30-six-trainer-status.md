# Six-trainer status at 12:25--12:26 CST

- Read-only Slurm/log checks: L1/399733, L2/399747, S1/399040,
  S2/397373, S3/397375 and S4/397376 are RUNNING. Two successive reads
  confirm all six global steps advance. No training setting or job changed.
- Latest sampled epoch/step: L1 263/292140, L2 182/202380,
  S1 143/290360, S2 212/431280, S3 203/414000, S4 206/419720.
  Budgets remain 500 epochs for LC and 600 for ablations; all S runs stage C.
- LC latest completed epoch action loss: L1 epoch262 0.0068693,
  270.18 samples/s; L2 epoch181 0.0088123, 104.50 samples/s.
  L2 preceding epoch throughputs 239.30,173.33,117.76,104.50 indicate a
  recent slowdown; cause unverified. S1 latest cumulative epoch throughput
  70.46 samples/s; S2/S3/S4 106.25/104.80/100.12. Sampled losses and
  gradients are finite. S action/utility losses: S1 .00937/.06062,
  S2 .00745/.05984, S3 .00777/.05464, S4 .00761/.05491.
- LC stderr empty; S stderr contains only the known eight-workers/two-CPU
  affinity warning. Every latest.ckpt exists; L1 retains epoch200 and
  S2/S3/S4 retain epoch200. This check did not deserialize checkpoints.
- Independent nvidia-smi checks (S1/S3 through their own overlapping Slurm
  steps): all H200/143771MiB; used memory L1/L2 29881MiB, S1 30163,
  S2 30101, S3/S4 30149, approximately21%. S buffers remain active.
  Momentary utilization L1/L2/S1/S2/S3/S4 92/35/14/24/35/51 percent;
  snapshots alone do not establish average utilization or slowdown cause.
- S2/S3 current allocations end October30 23:36:11/23:38:38. No dependent
  continuation currently appears in squeue. Future continuation was not
  exercised or changed by this status-only check.

## Requested S2/S3/S4 loss plot at 12:33 CST

- Plotted only `epoch_complete` sample-weighted means, without smoothing or
  partial-epoch points: S2 through212, S3 through203, S4 through206.
  Single PNG/PDF includes action and utility full histories plus epochs100+
  zooms. Complete epoch coverage was checked for each run.
- PNG/PDF, extracted `epoch_losses.csv` and checkpoint inventory `summary.json`:
  `/data/wangzihao/outputs/manifeel/analysis/s2_s3_s4_loss_20261030_123318/`.
  Source is each run's existing `metrics.jsonl`; no training state changed.
- All three checkpoint directories contain exactly `epoch1.ckpt`,
  `epoch6.ckpt`, `epoch10.ckpt`, `epoch20.ckpt`, `epoch50.ckpt`,
  `epoch100.ckpt`, `epoch200.ckpt`, and `latest.ckpt` at inspection.
  Latest checkpoint contents were not deserialized; `latest.ckpt` continues
  to be replaced by training.
