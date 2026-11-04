# Resume interrupted FTP pipelines and supervise future interruptions

- User requested resuming the two failed FTP jobs and automatic continuation
  when borrowed GPUs are reclaimed. Slurm recorded parent FAILED15:0 and
  derived SIGTERM0:15; the logs do not establish who sent the signal.
- Validated insert_hole latest.pt at step300/phaseA, with optimizer and all RNG
  states and matching training identity. Its 8,528 labels are complete.
  put_bottle has621 valid committed shards/4,968windows, including one shard
  newer than the last progress event. No labels were regenerated or discarded.
- Replacements: insert_hole423843 and put_bottle423844. Same immutable source,
  configs, output directories, authorized overload account, single H200 via
  gpu,gpux and48h common walltime. Other GPU jobs remain untouched.
- insert_hole423843 started on server13, GPU UUID
  GPU-99c2eeff-7137-8916-a621-b3ff30a8570e (Slurm physical index1), initially
  empty143771MiB H200. Restored step300 and reproduced original step320/340
  utility losses exactly. Existing independent VRAM holder follows the new
  registry ID; observed total39555MiB. put_bottle423844 is pending Resources.
- Added independent CPU recovery supervisor423864 for all seven native runs.
  Poll120s; terminal/accounting grace120s; retry SIGTERM/SIGKILL, node failures,
  preemption, timeout and incomplete normal exits. Preserve committed source
  identity and label/checkpoint contracts. Check live/pending continuations
  before submission, adopt matching output directories and lock supervisor/
  registry updates. Fail closed on scheduler errors; explicit pause,
  cancellation, OOM and current Python failures do not loop. Backoff up to30min,
  stopping after8retries with no durable progress. CPU wrapper requeues before
  its own5day walltime. Runtime snapshot and history are under the canonical
  registry's `recovery/` directory; training snapshots were not edited.
- Focused recovery tests:18passed. They simulate interrupted exits, missing
  accounting, scheduler outage, persisted retry/backoff, identity preservation,
  pending-job deduplication and continuation races. A live one-pass check
  found all seven existing GPU/pending jobs without making a duplicate.
- Full regression:386passed,2skipped in165.31seconds; shell syntax passes.
  CPU supervisor request permits backfill down to1hour from the5day maximum,
  with the same advance signal/requeue behavior at its assigned walltime.
- Supervisor423864 actually started22:20:12 onserver18 with a backfill
  allocation of1hour. Its first successful poll was22:20:24, with all seven
  native runs recognized and no scheduler errors or duplicate submissions.
