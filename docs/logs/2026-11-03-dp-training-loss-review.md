# DP training loss review

- Read the two production `train/progress.jsonl` logs and Slurm states without
  changing jobs. Tabero417432 is COMPLETED: E200/341000 updates, final epoch
  mean training loss0.000330275641. E25/E50/E100/E150 means are respectively
  0.005245267144/0.003405891306/0.001606833651/0.000657586885.
- UniVTAC417405 remains RUNNING on server23. Snapshot2026-11-03 19:16:58+0800:
  E350 complete,366920/524000updates (70.02%), E350 mean0.000205328321.
  E50 throughE350 checkpoints every50epochs exist. Last30min wall-clock pace
  including checkpoint overhead implies8.45hours remaining, approximately
  Nov4 03:44+0800 if pace holds.
- Generated PNG/PDF plots, epoch CSVs, frozen log snapshots, summary JSON and
  a reproducible Matplotlib script under
  `/data/wangzihao/outputs/loss_review_20261103_1916/`.
  Visually checked both PNGs. Tabero plots include raw windows/epoch means,
  saved checkpoints, position/force components and LR. All logged training
  window losses finite. These logs contain no validation or rollout results;
  decreasing training loss alone does not select the rollout Teacher.
