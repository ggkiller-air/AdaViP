# L1 restored training status at 12:35 CST

- User clarified the six requested tasks are ACT pull-drawer plus the five
  fresh November 1 runs B/C/D/E/input-dropout10. Original L1/A is only a
  historical comparison baseline and is not one of these six tasks.
- ACT job414991 completed November 2 11:54:03CST, ExitCode0:0, wall time
  01:17:34. COMPLETE.json confirms20000/20000updates; best validation loss
  .03061487811 at step19825 (L1=.02974586240); final validation=.03183896061.
  Initial validation=.17718587119. Best/last, fixed5k/10k/15k/20k snapshots
  and training_latest.pt exist. Final deployment archive:
  /data/wangzihao/outputs/openvitac_act/pull_drawer_100_seed0/handoff/final/ACTPullDrawer.tar.gz
- ACT's first job412676 failed before training (one second). Saved failure
  audit attributes this to the absent Slurm log parent; retry414991 succeeded.
  Successful job stderr contains only torchvision deprecation warnings.
  These validation losses are not rollout success rates.
- Read-only scheduler, full epoch metrics/progress, checkpoint inventories and
  GPU telemetry checks. All five jobs RUNNING, Restarts=0, stderr empty;
  progress advances between reads and logged action/future/total losses and
  gradient norms are finite. No training or scheduler settings changed.

| Group | Job | Current / completed epoch | Step | Last completed action loss | Future loss | Recent samples/s | Remaining hours |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| B tactile raw | 412402 | 214 / 213 | 237780 | .00885860818 | .000118659801 | 271.5 | 22.1 |
| C tactile latent | 412403 | 192 / 191 | 213860 | .00945885454 | .0115845931 | 246.9 | 25.8 |
| D vision+tactile raw | 412404 | 164 / 163 | 181720 | .01050005295 | .000662774625 | 229.6 | 30.5 |
| E vision+tactile latent | 412405 | 128 / 127 | 142080 | .01245245186 | .0144533669 | 201.5 | 38.2 |
| Input dropout10 | 412407 | 173 / 172 | 192140 | .01129166576 | none | 279.6 | 24.4 |

- Shared protocol is batch64/E500/seed42. Loss and speed come from metrics.jsonl;
  ETA uses the last ten completed epoch wall times and current progress,
  excluding future queue delays. Estimated completions: November 3 10:41,
  14:24, 19:07, November 4 02:46, and November 3 13:01 respectively.
- At matched E100, original A action=.01459620793. B/C/D/E action values are
  .01463783863/.01454273566/.01449803526/.01431765400: relative changes
  +.285%/-.366%/-.673%/-1.908%. Training loss does not establish rollout success;
  raw and latent future losses use different objectives and scales.
- Dropout E172 full-input loss=.01070005733; weighted missing-input loss=
  .01663989604; actual full-input share=90.04%.
- Every formal checkpoints/latest.ckpt exists; B retains E50/100/200,
  C/D/E/dropout retain E50/100. Original A retains E50/100/200/300/400/500.
  Checkpoint contents were not deserialized.
- Allocation-local monitoring shows H200 GPUs using approximately 28.1--28.7GB
  of 143771MiB. Independent SSH reads confirmed H200 telemetry on all three
  hosts; allocation device visibility on server13 exposes E's GPU for that
  SSH read, so B/dropout memory comes from their own monitor logs.
- E's estimated finish exceeds its November 3 22:56 allocation end. The frozen
  stage1_future.sbatch has USR1@300 safe-stop handling and submits an afterok
  continuation when SAFE_STOP.json exists, resuming latest.ckpt. This mechanism
  is configured but has not yet been exercised for these jobs. D's ETA is
  close to its November 3 21:02 limit and may also need continuation if slower.
- Source root:
  /data/wangzihao/outputs/manifeel/l1_restored_batch64_ep500_seed42_20261101/
