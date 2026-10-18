# Stage0 same-node DDP4 preflight and 1000-epoch continuation

- User requests four GPUs on one node, per-rank16/global64; retain LR8e-5
  and1,082updates/epoch. Verify cache sharing, weighted sampling, EMA,
  all-task validation aggregation and restart before replacing single job368155.
  New target1000epochs; retain early epoch snapshots then every100epochs.
- At15:27 inspected368155/server36: H200143771MiB/29885MiB used/82%
  utilization, sole allocated trainerPID3674478. Job still runs unchanged;
  interactive CPU job366142 also remains untouched. At15:41 writer-node
  progress wasstep13440, epoch13; no interruption requested.
- Added torchrun-dispatched workspace, preserving bare model/EMA/optimizer
  checkpoint keys and original global batch/config. One deterministic global
  WeightedRandomSampler plan per epoch is audited across ranks and split within
  each global batch; tail21 becomes6/5/5/5, with globally sample-weighted
  gradient scaling. Empty-rank dummies have zero training/statistical weight.
- DDP requires shared read-only NPY mmap cache. Rank0 verifies/prefaults first;
  remaining ranks verify the same file pages. Record file-inode identities and
  /proc/smaps NPY RSS/PSS to verify physical sharing. Spawn workers reattach
  file paths instead of serializing image arrays, avoiding NCCL+fork workers.
  Allocation request4H200/48CPU/512GiB; no fourfold279GiB private cache.
- Global training sums/counts and validation task/condition sums/counts are
  all-reduced without padded validation duplicates. Validation generates the
  original global-batch noise before slicing, uses EMA, and keeps all ten
  paired conditions. Full raw/EMA hashes and cursors checked across replicas.
- Checkpoints are rank0-only atomic writes with barriers, all-rank RNG,
  global sampler cursor, optimizer/EMA counters and global partial-epoch stats.
  Restore each rank's current CUDA device RNG, not GPU0 on every rank.
  Single-to-DDP migration retains states and gives other ranks independent
  reproducible streams; exact DDP restore rejects world-size changes.
- CPU/Gloo testing caught two real integration pitfalls before GPU submission:
  inherited zero-element device-anchor parameter was trainable but unused;
  freeze it locally without changing third_party or checkpoint shapes. DDP's
  automatic gradient-bucket rebuild changed floating reduction order after
  restart, affecting Adam on tiny gradients. Keep constructor bucket layouts
  fixed using find_unused_parameters=True; four-rank uninterrupted vs resumed
  raw/EMA/optimizer then passes bitwise comparison. This CPU result is not a
  guarantee across different CUDA algorithms/topologies.
- Separate bounded runtime stop controls from scientific epoch budget. Short
  run keeps original300-epoch scheduler, reaches two epoch boundaries, exits,
  resumes and executes120more updates. Assessment requires both phases,
  one full epoch, nine-task2060-sample paired validation, unchanged69205-sample
  plan/1082updates/tail, cache sharing, EMA/RNG/cursor restoration, and >=1.15x
  measured global throughput versus live single-GPU last20windows. Includes
  separate full-epoch wall time; does not label pure training throughput as
  end-to-end speedup. Assessment never signals or cancels the baseline.
- Explicit budget extension permits only num_epochs growth when authorized;
  all other scientific settings stay strict. Anchor future cosine to saved
  current LR, retain optimizer/EMA/global_step, avoid restarting warmup or
  jumping LR.1000epochs=1,082,000updates; parallelization alone never quarters
  the original per-epoch update budget.
- Long launcher requires passed measured report and explicit long-mode env;
  it never implicitly promotes a failed/unmeasured preflight. Early snapshots
  1/2/5/10/20/50 if not already passed, then100/200/.../1000; latest every1000
  updates and epoch, best on full-input EMA macro loss, interrupt on stop.
  USR1 warning is batch-only and writes a local stop-file so torchrun agent
  isn't killed before ranks can checkpoint. Long-mode time-limit exits alone
  request same-source afterok continuation. Existing single launcher untouched.
- Final focused regression66passed/86.02s, including single-to-four-rank
  migration, true four-process Gloo training, exact mid-epoch model/EMA/Adam
  restore and continuation, no-duplicate validation, unequal/empty tails,
  spawn-worker mmap reattachment, LR extension and eight fail-closed assessment
  cases. Full repository200passed/1skipped/1known missing-FiLM-CLIP-asset
  failure in104.08s. Persistent logs `ddp_final_tests_20261017.log` and
  `ddp_repository_tests_20261017.log` under the Slurm logroot. Shell syntax,
  source SHA256 and whitespace checks pass; no third_party modifications.
- Frozen preflight run:
  `/data/wangzihao/outputs/manifeel/full_sensor_ddp4_batch16_global64_preflight_20261017_154556`.
  Copied a stable atomic single-GPU checkpoint (epoch13 completed,step14066,
  EMA14066,cursor0) into seed.ckpt; checked real payload/provenance and exact
  scientific-config compatibility on CPU before requesting GPUs. Old
  protocolca774c8c... migrated to7b2bebe6... solely because checkpoint retention
  settings are now explicitly operational, not scientific. Dataset and
  normalizer fingerprints unchanged. Source, seed SHA256, Git metadata and
  seed_metadata.json retained with the run; third_party reference read-only.
- Submitted368498 at15:48:41, accountyangyc/partitiongpu, one-node4H200,
  48CPU/512GiB,90min request/min40min backfill. At15:49 pendingPriority,
  no allocation or measured NCCL/GPU throughput yet; QoS has no reported
  MaxTRESPU/MaxTRES/GrpTRES limit. Do not report CPU tests as GPU acceleration.
  Short job checks two complete epochs14/15, then real restart+120updates,
  and writes ddp_preflight_report.json. It does not cancel368155 or submit
  long training by itself. Actual long handoff remains gated on its result.
- At15:49 writer-node single-GPU progress15040, epoch14/cursor974;
  stillRUNNING. Recent sampled training throughput187.5/s (earlier~270/s),
  so assessment uses live recent20window harmonic throughput rather than
  assuming the earlier figure is constant. Current baseline ends20:30:34
  and its existing safe300-epoch continuation mechanism remains intact.
  After a passed short test, keep single running until long resources are
  ready, safely stop it without triggering its USR1 auto-continuation,
  and migrate its then-final checkpoint to authorized1000epochs. Never
  roll back to this preflight's older seed or branch.
- At16:19 checked actual results:368498 ran15:53:53–16:18:39 onserver03,
  COMPLETED0:0. Two full epochs14/15 plus true120-update restart completed.
  Cache279GiB logical and total-rankPSS279GiB confirms physical sharing;
  validation9tasks/2060samples, raw/EMA hashes, cursor and per-rank RNG passed.
  Original window-only report said2.1197x (327.48/s versus momentary154.49/s).
  Post-run review must not treat that as sustained training acceleration:
  full epochs averaged269.15/s versus the single's matching epochs235.42/s,
  only1.14328x, below the established1.15gate. DDP epoch wall318.60/311.66s;
  first windows each epoch took~40s with91–95% data wait (worker startup).
  Single had recovered~270/s local windows by this check. No end-to-end2x claim.
- Tightened assessment and long-launch guard to also require matched full-epoch
  training speedup including loader startup. Preserved original report as
  ddp_preflight_report_window_only.json; reran assessment with reviewed code,
  authoritative ddp_preflight_report.json nowpassed=false, solely for full-epoch
  speed. Added slow-epoch regression; assessment9passed/13.87s. The original
  frozen training source is unchanged; updated reviewer source belongs to the
  follow-up audit, not retroactively to the completed GPU job. Single368155
  remainsRUNNING (epoch20/step21100 at16:19); no1000-epoch DDP job submitted.
  Next optimization should address per-epoch worker recreation and remeasure
  full-epoch speed before any replacement. Same-node does not pin a hostname:
  all eligible four-H200 nodes may schedule it. Multi-node requires a separate
  launcher and node-local cache/RNG/aggregation checks; current code intentionally
  rejects it, rather than assuming that a NAS mount shares RAM across nodes.
