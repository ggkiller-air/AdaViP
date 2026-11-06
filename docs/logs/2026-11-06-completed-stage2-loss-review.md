# Completed Stage2 loss review

- Read the completed ManiFeel DP, Tabero DP and native Tabero-VTLA production
  logs and retained checkpoint names; no jobs or weights changed. None has a
  named best checkpoint, and these logs contain no validation/rollout scores.
- Generated four PNGs in `/public/home/wangzihao/projects/`: the combined
  `uvip_three_runs_loss.png` and `{manifeel_dp,tabero_vtla,tabero_dp}_uvip_loss.png`.
  Reproducible script, curve/checkpoint CSVs and summary JSON are in
  `uvip_loss_review_20261106/` beside them. Visually inspected combined figure.
- DP uses logged sample-weighted complete-epoch means (600/600 and 200/200
  epochs verified). Native VTLA uses 50 logged batches in the preceding 1000
  steps within the same phase; these are sampled every20steps, not every update.
  Require the full window for checkpoint ranking. Exclude phase A action zeros
  and rank only retained B/C checkpoints; this is not EMA or held-out evaluation.
- Lowest saved action/utility losses: ManiFeel E600 0.002227378675/0.034133057647;
  Tabero DP E200 0.000332391070/0.061904818018. VTLA action minimum at step32500
  0.007385356631; utility minimum at step12500 0.265277400315. Final step50000
  action/utility 0.007660072455/0.382914970815. Metrics cannot be compared directly
  across backbones. All parsed plotted values finite; zero malformed log lines.
