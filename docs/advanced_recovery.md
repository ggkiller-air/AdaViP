# Native UViP interruption recovery

The GPU pipelines already resume atomic label shards and Student checkpoints.
`adavip.advanced.recovery` runs in a separate CPU allocation so it can submit
a replacement even when the GPU allocation and its shell have been killed.
It follows the seven native runs in `UVIP_RUN_REGISTRY`; it does not modify
the frozen training source, configurations, Teacher, labels or running jobs.

The supervisor checks Slurm every 120 seconds. Once a run has disappeared
from the queue, it allows at least 120 seconds for accounting and any normal
continuation to appear before resubmitting. It retries SIGTERM/SIGKILL exits,
preemption, timeout, node/boot failures, and incomplete successful soft stops.
The replacement uses the same source, output, configuration and account, with
one H200 requested through `gpu,gpux` and the common partition walltime.

Existing running, pending, requeued and completing jobs prevent a replacement.
The supervisor also adopts a normal continuation after matching its output
directory, closing the gap between `sbatch` and registry updates. A registry
lock protects updates; a supervisor lock prevents two recovery loops, and the
existing GPU `run.lock` prevents concurrent writers. Slurm query errors defer
the decision instead of submitting a job.

Retries without durable progress back off to 30 minutes and stop after eight
consecutive replacements without a new checkpoint or label artifact. Python
exceptions recorded for the current job, OOM and explicit cancellations stop
automatic retry. This keeps broken configurations from consuming allocations.
Create `RECOVERY_PAUSE` in a run's output directory **before** cancelling its
GPU job to keep that run paused. Remove it to allow the supervisor to reconsider
the existing job state; explicit Slurm cancellation still requires a deliberate
manual resubmission.

CPU deployment uses a separate management-code snapshot:

```bash
sbatch --job-name=uvip_recovery \
  --output=/path/to/recovery/supervisor_%j.out \
  --error=/path/to/recovery/supervisor_%j.err \
  --export=ALL,UVIP_RUN_REGISTRY=/path/to/registry.json,UVIP_RECOVERY_SOURCE=/path/to/management_source \
  /path/to/management_source/slurm/advanced/recovery_watch.sbatch
```

The CPU request allows a one-hour minimum for backfill, up to five days, and
the wrapper requeues itself before the assigned walltime expires. Recovery
status and retry history are written beside the registry under `recovery/`:
`status.json`, `events.jsonl`, per-run JSON states and `ERROR.json` on scheduler
errors. `COMPLETE.json` ends recovery for a run. SIGKILL can lose updates since
the most recent atomic checkpoint; it does not invalidate earlier checkpoints
or committed label shards.

On 2026-11-03 the interrupted FTP runs were resumed using their original source
identity: insert_hole from step300 (full optimizer/EMA/RNG state), and
put_bottle_in_shelf from 621 committed shards containing 4,968 label windows.
The former was verified to reproduce the original step320/340 utility losses.
