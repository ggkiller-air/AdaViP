# Stage0 training status and loss curves

- Read-only audit at2026-10-18 20:28–20:31+08:00: job370163 remainsRUNNING
  onserver30,1H200/24CPU/384GiB. Scheduler confirmsTimeLimit=48h andTimeMin=48h,
  start2026-10-18T03:13:49, end2026-10-20T03:13:49. The scheduling helper
  succeeded at03:09:49; no further six-hour slicing is required by this allocation.
  Sampled GPU143771MiB total/29885MiB used/65% utilization. Left all jobs intact.
- Current run:
  `/data/wangzihao/outputs/manifeel/full_sensor_reference_batch64_ep300_seed42_decoded_localipc_20261017_142753`.
  Completed275epochs; progress at20:30 reports epoch276,step298400,cursor850/1082,
  159.0samples/s and0.22%data wait. Actual saved budget remains300epochs,
  not1000; no training setting or live immutable source changed in this audit.
- Generated `plots/stage0_loss_curves.png` and its JSON summary in that run.
  Snapshot at20:29:42 uses275complete epoch records, no smoothing. Train loss
  at275is0.0055904136; full-input EMA validation task-macro0.1046940624,
  sample-micro0.0808276234. Best task-macro remains0.0278735495 at18.
  Strong train/validation divergence is consistent with overfitting, not proof
  of simulator performance. Mixed-input raw train and full-input EMA validation
  differ in conditions; labels explicitly distinguish them.
- CPU memory-mapped deserialization verified `best.ckpt`:epoch18/step19476;
  `latest.ckpt`:275completedepochs/step298000/nextbatch450;
  `interrupt.ckpt`:135completedepochs/step146231/nextbatch161.
  Each includes model,EMA,optimizer and workspace metadata. Sixteen complete
  checkpoints total~75GiB:best,latest,interrupt andepoch20through260every20.
  `.partial` files are in-progress atomic saves, not evaluation inputs.
- Existing checkpoints can be evaluated without waiting for training completion,
  but the old stock evaluation entrypoint cannot directly load this run: it
  requires `cfg.task.env_runner`, absent from Stage0 config. The old multitask
  command also aggregates `test/success_rate`, unsuitable for canonical reporting.
  A Stage0 evaluation configuration/entrypoint and GPU smoke remain necessary.
  Checked actual encoder: language expects[B,To,512], compatible with the
  existing task-conditioned adapter; do not invent a[B,512] mismatch.
  Use saved EMA+normalizer, all3cameras/2TacRGB, matching image sizes/action
  truncation. Recommend smoke first, then50env/task/500steps, report
  `test/mean_score`. No simulator evaluation was launched or success rate claimed.
- Added standalone `scripts/manifeel/plot_full_sensor_training.py`; completed-line
  parsing tolerates a live partial tail and deduplicates resumed epochs. Three
  CPU tests pass in1.54s, including PNG output and best-vs-latest selection.
  Visually inspected the180dpi PNG. No old experiments or third_party changed.

Reproduce the figure:

```bash
/public/home/wangzihao/.local/miniforge3/envs/manifeel/bin/python \
  scripts/manifeel/plot_full_sensor_training.py \
  /data/wangzihao/outputs/manifeel/full_sensor_reference_batch64_ep300_seed42_decoded_localipc_20261017_142753
```
