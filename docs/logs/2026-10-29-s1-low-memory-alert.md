# S1 job399040 low-memory-utilization alert

- Read-only check after the user's14:56:07CST alert: `scontrol show job
  399040` reports RUNNING on server23, S1/T2/batch32, oneH200,
  32allocatedCPUs,384GiB hostRAM. No job/configuration was changed.
- Alert reason is `low_mem_util`:7619/143771MiB GPU memory,5.30%.
  Its25% GPU utilization and14% SM utilization are distinct from memory
  occupancy. `idle_minutes=180` is the monitor's classification; it does not
  establish180minutes without training progress. The monitor implementation
  and action at `threshold_minutes=360` were not available in this check.
- Independent SSH/nvidia-smi check confirms H200,7619MiB occupied,18% GPU
  utilization at sampling, and trainerPID3697198 using7604MiB. Stdout advanced
  from epoch52/global_step104740 to104880 during inspection, approximately
  105samples/s, action_loss0.0152,utility_loss0.0669. Stderr has a DataLoader
  CPU-affinity warning, with no traceback. Latest checkpointmtime15:08:43;
  epoch50 checkpointmtime14:58:01 on2026-10-29.
- Despite32allocatedCPUs, trainer affinity is0,96 and sampled workers are
  bound separately to0or96: the known step-affinity issue also affects S1.
  This is a performance limitation requiring independent measurement before
  changes; low memory occupancy alone does not establish a training fault.

## Authorized minimum-memory occupancy mitigation

- User confirmed the administrator requires at least15% GPU memory occupancy
  and explicitly authorized loading extra data while continuing training.
  S1 freezes image encoders; S2 trains them; S3/S4 have separate frozen context
  and trainable Action branches (`stage2_policy.py`). Before mitigation,
  S1/S2/S3/S4 occupied7619/17031/17593/17591MiB, respectively; even S2--S4
  were below15% of143771MiB. Their larger use did not satisfy the stated rule.
- Started one independent overlapping Slurm step inside each existing job,
  without restarting trainers or changing models, batch sizes, optimizers,
  precision, checkpoints or process affinities. Each step requests oneCPU,
  2GiB hostRAM and its existing GPU with `srun --jobid=ID --overlap --exact
  --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=2G --gres=gpu:1`.
- Helper uses CUDA runtime directly to allocate and zero a buffer, then sleeps
  60seconds between occupancy/liveness checks. Target21%, refill below19%;
  preserves32GiB free on allocation and releases buffers if free memory falls
  below16GiB. No recurring GPU compute or dataset/NFS reads. This buffer does
  not improve training efficiency. Per-job file locks prevent duplicate
  helpers; helpers exit after the original trainerPID disappears and are also
  contained by the existing allocation's lifecycle.
- Verified with nvidia-smi from inside each corresponding allocation at15:36:
  S1/39904030159MiB(20.98%); S2/39737330099MiB(20.93%);
  S3/39737530149MiB(20.97%); S4/39737630149MiB(20.97%). All four
  helpers loaded successfully, with empty stderr. HelperPIDs are576753,
  302766,302765,3311103; original trainerPIDs3697198,2694457,2696789,
  2089872 remain unchanged. The independent Transformer allocation was untouched.
- Exact helper source, launcher command/PID, stdout/stderr and GPU verification
  are under each `sN/memory_guard_JOBID/` directory in the existing ablation
  root `/data/wangzihao/outputs/manifeel/stage2_encoder_ablation_t2_batch32_ep600_seed42_retry_20261028_233606/`.
  These are runtime artifacts outside Git; frozen scientific sources remain
  unchanged. Meeting the measured occupancy floor does not independently
  verify how the cluster monitor resets its prior alert counter.
- At15:36:33 all four jobs remain RUNNING and steps advanced after buffers
  loaded: S1 106520to106660; S2 184000to184140; S3 174580to174740;
  S4 182060to182200. Reported throughput remains approximately102--104samples/s
  with finite action losses. Per-job `training_verification.json` records this
  check; no lasting throughput improvement is claimed.
