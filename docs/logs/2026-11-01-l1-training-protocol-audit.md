# Original L1 versus Future and input-dropout training protocol

- Read-only audit of all six actual `config.yaml`, `COMPLETE.json` and
  `metrics.jsonl` files. Original root:
  `/data/wangzihao/outputs/manifeel/language_conditioning_t2_batch64_ep500_seed42_20261028_204459/l1/`.
  Future root: `stage1_future_k4_batch32_ep100_seed42_20261031_133616/{B,C,D,E}/`;
  dropout: `l1_dropout10_and_mask_batch32_ep100_seed42_20261031_170151/l1_dropout10/`
  under the same output parent.
- Original: batch64, E500,1114updates/epoch,557000total. New five:
  batch32, E100,2228updates/epoch,222800total. All use AdamW8e-5,
  betas(.95,.999), eps1e-8, weight decay1e-6, warmup500updates,
  total-update cosine decay, gradient clip1, seed42 and policy EMA.
- Same base LR does not mean the same schedule: using the implementation's
  last-update index, original E100 LR is approximately7.24138e-5;
  new E100 LR is approximately3.99439e-15. New training processes20% of the
  original sampled examples and40% of its optimizer updates.
- Original fixed snapshots: E50/100/200/300/400/500; new: E20/40/60/80/100.
  Both save rolling latest every1000updates and each completed epoch, plus
  startup/safe-stop saves. Fixed files are preserved; no validation-best
  selection is used in these workspaces.
- Original action loss E100=.01459621, E200=.00918835, E500=.00250800.
  New final B/C/D/E=.01246286/.01216929/.01221757/.01205265;
  dropout=.01404156 (includes missing-input samples). These training losses
  do not establish simulator performance or identify the cause of reported
  poor performance. Batch/budget/schedule differences confound objective
  ablations; saving cadence itself does not change optimizer updates.
- No training, evaluation, job changes or configuration edits performed.
