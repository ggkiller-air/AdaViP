# L1 cache startup: memory allocation diagnosis

- At13:24 L1/399171 had spent approximately26minutes on server51. This is
  existing float32.npy cache loading/copying/hashing, not image decoding. Only
  two tiny array completion events were emitted, but in-flight array pages
  were being read: process read_bytes43,712,765,952 and RSS roughly60--62GiB.
  Completed-array logs do not report partial array progress.
- Three-second process sample:200.1%CPU, with a subsequent sample showing
 199.8%system CPU and0%user CPU. This is kernel work rather than SHA256/user
  computation. The trainer is restricted to CPUs16,112, sibling threads of
  one core; its step cpuset allows16--31,112--127. No process was adjusted.
- During the second3second sample, host compact_stall and compact_fail each
  increased96, compact_success0; THP fault fallbacks31 and allocations0.
  Global memory-pressure full avg60 was21.65%. These are host-wide counters,
  so association with this process alone is not proof of causality.
- THP enabled=madvise and defrag=madvise. L1 anonymous huge pages were
  approximately38.2GiB. Its job is below the384GiB hierarchical memory limit,
  with memory.failcnt0 and no process swap. Both NUMA nodes are allowed; no
  strict one-node cpuset memory restriction was found. Node1 free memory was
  approximately2.75GiB, while the host has substantial reclaimable file cache.
- Kernel documentation describes direct reclaim/compaction stalls for advised
  hugepage allocations and the meanings of the compact_* counters:
  https://docs.kernel.org/admin-guide/mm/transhuge.html . NumPy documentation
  provides process-local import-time NUMPY_MADVISE_HUGEPAGE=0 to disable its
  large-array hugepage requests:
  https://numpy.org/doc/1.24/reference/global_state.html . Installed NumPy1.23.3.
- Added bounded standalone CPU allocation probe, avoiding dataset I/O and GPU
  use. It compares alternating fresh processes with NumPy hugepage requests
  off/on/off/on; each fills/hashes two1GiB arrays on one physical core and has
  a40second deadline. Job399246 runs on server51 with its own8CPU/32GiB allocation
  and10minute limit. It changes no sysfs setting or existing training process.
  Artifacts: `/data/wangzihao/outputs/manifeel/cache_allocation_probe_20261029_132832/`.
  The first hugepage-off case completed both arrays in1.74seconds; further
  measurements are pending at the initial log entry. No production change is
  justified until the controlled on/off comparison is checked.

## Completed same-node control

- CPU-only job399246 completed successfully in1:27 on server51. Alternating
  fresh processes on the same reserved physical core produced:
  hugepage-off two1GiB arrays fill+hash1.737s; hugepage-on timeout40s;
  hugepage-off1.732s; hugepage-on timeout40s. Including interpreter startup,
  off cases took2.61/2.47s and on cases exceeded40s. Zero-array SHA256 values
  match between off repeats. This reproduces a severe process-local THP-related
  startup slowdown without dataset/NFS reads or changes to production jobs.
- Together with L1's kernel-only saturated CPU and node compaction failures,
  this identifies hugepage allocation/compaction as a strongly supported
  contributor. It does not quantify all NFS, affinity or full-cache effects,
  or establish the resulting279GiB load time after mitigation.
- Future repository LC batch wrapper now defaults
  `NUMPY_MADVISE_HUGEPAGE=0` before Python/NumPy startup (explicit overrides
  remain possible). Shell syntax check passes. Existing immutable sources,
  global THP sysfs settings and current L1/L2/other trainers are unchanged.
  NumPy reads this setting at import, so it cannot repair an already-running
  process merely by changing a shell variable. Restart of L1 has not been done,
  honoring the user's instruction to leave running jobs untouched.

## Authorized L1 restart

- User explicitly requested restarting L1 after reviewing the THP comparison.
  Pre-stop inventory: server51 H200,143771MiB total,30927MiB used,0%GPU,
  trainerPID2564265. Latest checkpoint remains epoch0/batch3/global_step3,
  before cache loading. New NumPy process confirms its internal
  `_get_madvise_hugepage()` flag is False with the environment override.
- Created immutable source under the original LC root's
  `resume_no_thp_20261029_134717/source`. All596source files match the previous
  IPC retry except the LC batch script's NumPy environment setting. Runtime
  source fingerprint remains `5a49934b...658a2`, exactly matching checkpoint
  provenance. Hard-linked original latest into the restart directory; no
  checkpoint mutation/migration or scientific-setting change.
- Cancelled only L1/399171, ending13:47:40 after49:37; its cache had not finished.
  Submitted replacement399298 as one gpu,gpux job at the queried common maximum
 48hours, Requeue=1 and `NUMPY_MADVISE_HUGEPAGE=0`. An afterany dependency on
 399171 plus the existing per-run lock prevents overlap during old-step cleanup.
  Submission and restart verification records are beside the new source.
- L2/S1--S4 and the existing Transformer remain running, unmodified. Initial
  replacement state is PENDING/Dependency; no new full-cache timing is claimed.
