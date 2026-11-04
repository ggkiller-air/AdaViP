# Current ablations versus UniVTAC Transformer and Stage0

- Read-only comparison around15:43--15:46. Existing production allocations,
  precision and training settings were unchanged. Current S1--S4 last three
  complete-epoch median samples/s:103.90/106.86/105.15/106.43. Each uses65265
  windows per epoch; corresponding measured train times628.17/610.78/620.71/
  613.21seconds, excluding post-epoch checkpoint writes. L1/L2 report about
  108/103samples/s; recent whole-epoch wall times are about673--678/708--711s.
- Existing UniVTAC Transformer/398812 reports145.59samples/s at epoch404,
  step878000. Its metric is cumulative since `fit()` entry, including logging
  and checkpoint pauses; the ablation metrics accumulate per-epoch timed
  batches. Thus approximately1.4x more time per sample for current runs is an
  observational comparison, not a controlled architecture benchmark.
- UniVTAC Stage0 root
  `/data/wangzihao/experiments/univtac/full_sensor_dp_batch64_ep600_seed42_20261019_093221/`:
  last283train windows median384.87samples/s. Last20epoch completion intervals
  have median215.5seconds and350.55samples/s, including between-epoch gaps and
  saves. Each epoch75543samples, batch64, explicitly BF16 autocast.
- Separate AdaViP Stage0 all-data run at
  `/data/wangzihao/outputs/manifeel/full_sensor_reference_batch64_all_data_ep600_seed42_20261018_233410/`:
  last20epoch medians227.56pure-train samples/s,313.17train seconds,
  338.01epoch-wall seconds. Earlier held-out-split ep300run median180.58
  samples/s over its last20epochs. Distinguish these from UniVTAC Stage0.
- Verified sources differ: UniVTAC uses four image streams(head,wrist,bilateral
  tactile), compact uint8 transfer and GPU normalization; current AdaViP uses
  five streams and float32 decoded input. UniVTAC Stage0 uses BF16 while current
  L1/L2/S1--S4 loops are FP32; batches are64for Stage0/LC and32for Stage2.
  These are different datasets, architectures and measurement scopes, so no
  single-factor speedup is established from their throughput ratio.
- Independent live `/proc/3483944/status` onserver13 confirms original
  Transformer affinity80--95,176--191(32logicalCPUs). Current ablation trainers
  retain the previously verified two-sibling-thread binding. Prior separate
  72-window/80-step affinity experiment improved S2 only0.8%,S3 15.2% with
  eight workers; do not claim the affinity defect explains the full gap.
  Additional bookkeeping/CPU synchronization differs but its contribution
  is unmeasured. The memory buffers were added after the slowdown already
  existed and perform no recurring GPU compute after initialization.

## L1/L2 versus AdaViP Stage0, user's corrected comparison

- User narrowed the comparison to AdaViP Stage0 and L1/L2; UniVTAC is outside
  the requested performance target. Saved configs confirm identical batch64,
  T2, horizon16, five native-resolution image streams, float32 decoded RAM
  cache, ResNet18/GroupNorm backbones and DP/U-Net width. L1/L2 mainly change
  small fusion/language heads(approximately263k parameters in313M models).
  This supplies no architectural justification for accepting a2.1x training
  time difference as normal, although exact228samples/s is not guaranteed.
- Both metrics sum timed training batches and exclude post-epoch diagnostics
  and saves. L1/L2 ordinary fixed-loss diagnostics take approximately4.7--5.4s
  per epoch, compared with approximately660--690s pure training. They cannot
  explain the approximately2x difference in the pure-training metric.
- Verified a runtime discrepancy in both immutable LC entrypoints: explicit
  `cudnn.deterministic=True`, `cudnn.allow_tf32=False`,
  `cuda.matmul.allow_tf32=False`. The original Stage0 entrypoint/workspace
  does not explicitly set these flags; historical live runtime flag values
  were not recorded, so TF32/determinism is a concrete candidate to measure,
  not a confirmed cause or a permission to change current scientific settings.
- Live LC affinities: L1PID828268/server30 limited to58,154;
  L2PID2840172/server27 limited to1,97. Both remain bound to sibling threads
  of one physical core despite32allocatedCPUs. Current H200 samples show
  L1/L2 memory30939MiB and100%/46% utilization, respectively; snapshots do
  not by themselves quantify bottlenecks. No running job was modified.
