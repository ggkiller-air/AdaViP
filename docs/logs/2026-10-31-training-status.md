# Training status at00:20--00:25CST

- Read-only scheduler, metric, checkpoint metadata and overlapping GPU checks.
  No job, model, runtime setting, queue or checkpoint was changed. Single-task
  twelve-run suite:10RUNNING,2PENDING/AssocGrpGRES (USB two-input404624/404625).
  Running jobs all passed their isolated40-update preflight; stderr is empty.
- Completed epochs / last full-epoch train loss at the recorded snapshot:

| Task | Inputs | No dropout | Drop40 |
| --- | --- | --- | --- |
| Plug | five | 158 / .035881 | 158 / .039446 |
| Plug | two | 225 / .028476 | 223 / .029755 |
| USB | five | 103 / .042581 | 103 / .045077 |
| USB | two | pending | pending |
| Sorting | five | 97 / .026658 | 4 / .087016 |
| Sorting | two | 88 / .031510 | 89 / .032294 |

- Budget remains400epochs, train/validation batch8, state tokenizer included.
  Typical recent throughput: five42--46samples/s, two64--66. One partial first
  window per epoch includes worker startup and must not be diagnosed as a
  steady slowdown. Sorting five/drop started only around00:13 and is behind
  because of its queue wait; its paired models are not at matched epochs yet.
- Complete-input EMA validation: Plug five E150 .188232/.192528, two E220
  .244738/.299083 (nodrop/drop); their E10 best values are .097239/.096507 and
  .095725/.096211. USB five E100 .103426/.106030 versus E20 best
  .068142/.067132. Training loss declines while held-out loss rises: possible
  overfitting/generalization risk, with only one held-out episode per task.
  These losses are not simulator SR; no new-run SR result is available.
  Sorting two E80 .038374/.034606; five nodrop E90 .040464, drop not yet E10.
- Each running teacher has latest.ckpt. Highest fixed snapshots: Plug five150,
  Plug two200, USB five100, Sorting five nodrop50, Sorting two50. Newly started
  Sorting five/drop has latest only. Early best.ckpt files are preserved.
- S1/399040 RUNNING, stageC, current epoch197/600, step401220 on second read;
  last complete196 action .00778299/utility .05800114,111.92samples/s. Last5
  complete epochs approximately112samples/s. Fixed snapshots through100;
  mmap metadata of latest confirms next epoch197/cursor0/global_step399840.
  Existing eight-workers/two-CPU affinity warning persists; no new exception.
- L1/399733 RUNNING, current epoch415/500, step461660 on second read; last
  complete414 action .00304043,267.59samples/s. Last5 epochs267--278samples/s.
  Fixed snapshots through400. mmap checkpoint metadata confirms completed414,
  global_step461196 and EMA step461196. Stderr empty. At current approximately
  276seconds/epoch, remaining86epochs take roughly6--7hours, not a guarantee.
- Independent GPU reads: all active jobs use H200143771MiB. L1 used29881MiB,
  S1 used30163MiB; teachers five32853--32873MiB and two31129--31195MiB.
  All exceed15% occupancy; CPU supervisor404638 is RUNNING and logged all ten
  teacher holders. Instantaneous utilization varied0--79%; step advancement,
  rather than one GPU snapshot, confirms progress.
- Other former jobs are not running: Slurm records S2/S3/S4 canceled by UID2116
  at October30 19:03, L2 canceled by UID2116 at21:17. This status check did not
  cancel or resume them and does not infer the cancellation reason.
- Evidence: `/data/wangzihao/outputs/manifeel/analysis/training_status_20261031_0020/`
  contains timestamp, scheduler listing and detailed single-task/S1/L1 JSON.

## GPU-memory occupancy recheck at00:37:41CST

- Independently checked all14 active training allocations. H200 total143771MiB;
  L1=29881MiB(20.78%), S1=30163MiB(20.98%); six five-input runs32853--32873MiB
  (22.85--22.87%), six two-input runs31129--31201MiB(21.65--21.70%).
  All exceed the administrator-requested15% threshold. USB two-input404624/
  404625 have now started, so all twelve single-task jobs are RUNNING.
  Memory-reservation CPU supervisor404638 remains RUNNING. No setting changed.

## Status recheck at11:00--11:03CST

- Slurm and metrics agree that11/12 single-task runs completed400epochs,
  ExitCode0:0. Every completed run has nonempty epoch400.ckpt and best.ckpt;
  all twelve teacher stderr files are empty. Sorting five/drop404588 remains
  RUNNING, completed360/current361, train loss.00611817,44.46samples/s;
  checkpoint updated at step214000. Estimated remaining roughly70--80minutes
  at recent speed. Scientific settings remain batch8/400epochs.
- Final train loss nodrop/drop: Plug five.014779/.019647, two.018016/.020860;
  USB five.015706/.019190, two.018446/.018145; Sorting five nodrop.005027,
  two.005675/.007020. Dropout losses include missing-input conditions.
- Full-input EMA validation best/last: Plug five nodrop.097239/.279180,
  drop.096507/.294295; two nodrop.095725/.326062, drop.096211/.408252.
  USB five.068142/.193032 and.067132/.201794; two.066783/.180806 and
  .068515/.218292. Sorting five.028597/.097568 and.028068/.093160(E360);
  two.029981/.106878 and.030268/.104609. All best epochs are10--50.
  Held-out losses rise substantially after early minima; no new simulator SR
  has been measured. Retain both best and late checkpoints for evaluation.
- L1/399733 COMPLETED at07:09:02,500/500,ExitCode0:0,557000steps, final
  action loss.002507999,266.32samples/s; epoch500/latest files exist,stderr
  empty. S1/399040 stageC RUNNING,completed255/current256,step521580;
  complete-epoch action.00632435,utility.05715856. Recent five epochs
  77--91samples/s, slower than previous approximately112, with217--267s
  DataLoader wait per epoch. Existing worker/CPU warning only; process keeps
  advancing. Epoch200/latest exist. At recent speed approximately70--73h
  remain, exceeding current allocation endNov2 15:50; source sbatch already
  handles safe-stop/checkpoint/dependent continuation. No runtime changed.
- Fresh independent GPU checks: active S1 H20030163/143771MiB(20.98%),
  Sorting five/drop32853/143771MiB(22.85%), both above15% including existing
  reservation. CPU reservation supervisor404638 remains RUNNING.
- Evidence: `/data/wangzihao/outputs/manifeel/analysis/training_status_20261031_1100/`.
