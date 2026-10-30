# Stage2 infrastructure benchmark

The current S2--S4 jobs reserve 32 logical CPUs but their trainer and eight
DataLoader workers share the two SMT threads of one physical core. Existing
training and queued production jobs must remain unchanged while independent
benchmarks measure the performance impact.

## Reproducible comparisons

`scripts/manifeel/benchmark_stage2_infra.py` tests S2 and S3 independently. S4
shares S3's dual-encoder architecture. All cases retain the actual model sizes,
batch32, T2, action horizon16, C-stage loss, clipping, AdamW and EMA. Every case
starts from the same E500-initialized weights and seed within its strategy.
Learning rates follow a representative C-stage cursor130000. Ten warmup steps
precede80timed steps; wall timing is bracketed by CUDA synchronization. CUDA
memory peaks, loss/gradient finiteness and phase times are recorded separately.

The input pool contains72real K64 windows, eight per task, copied once to RAM.
This isolates steady CPU collation/transfer and GPU compute without rereading
279GiB for each test. It does **not** measure whole-dataset RAM/NUMA behavior,
long-run convergence, checkpoint I/O or a complete training epoch.

Cases compare:

- One physical core/two SMT threads with8workers, matching the observed fault.
- All allocated CPUs with8,4or0workers.
- Deferred CPU metrics and foreach EMA, separately, reducing host synchronizations
  or small-kernel launch overhead while retaining the same losses.
- A GPU-resident batch, showing a compute bound without DataLoader/H2D costs;
  this repeats one batch and is not a valid formal training replacement.
- Matmul TF32 plus cuDNN autotuning; channels-last Conv2d weights; BF16 autocast;
  fused AdamW; and U-Net-only `torch.compile`, applied incrementally.

The baseline uses matmul TF32 off and cuDNN TF32 on, matching installed PyTorch
2.4defaults. Numerical alternatives are benchmark-only. Finite losses over80
steps are insufficient to establish equivalent convergence or benchmark scores.
Compilation is lazy: `warmup_seconds` includes cold compilation, while
`setup_seconds` only measures wrapper creation. A profiler trace and operator
table are collected after timing the corrected-CPU baseline.

Every completed case is atomically saved to the isolated output's `report.json`.
Signals/stop markers preserve completed cases; a requeued job repeats only
unfinished cases from the same fixed initialization. No production checkpoint
is written or training process adjusted. Separate stdout/stderr and a5-second
GPU monitor are retained per allocation. The benchmark requests one separate
H200,32CPUs,64GiB and a bounded one-hour allocation.

## Official references and interpretation

- [Slurm sbatch](https://slurm.schedmd.com/sbatch.html#OPT_partition): a comma
  separated partition list allows a single job to start in either partition.
  One job has one walltime; the submit helper queries current limits and uses
  their common maximum, currently48hours, with safe continuation for long runs.
- [Slurm srun](https://slurm.schedmd.com/srun.html#OPT_cpus-per-task): the step
  CPU request defaults to one CPU per task; inherited binding can confine
  multithreaded work. The independent benchmark explicitly supplies
  `--ntasks=1 --cpus-per-task=$SLURM_CPUS_PER_TASK --cpu-bind=cores` and checks
  its effective affinity before doing GPU work.
- [PyTorch tuning guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html):
  worker tuning, pinned/nonblocking transfers, NUMA locality, channels-last/AMP,
  avoiding host synchronization and compilation are relevant candidates.
- [PyTorch AdamW](https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html):
  fused and foreach implementations can reduce optimizer overhead. The installed
  2.4.1build supports the tested fused constructor and foreach EMA primitive.
- [NVIDIA training performance guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-getting-started/index.html):
  Tensor Core efficiency depends on precision, operation dimensions and layout.
  The current convolutional/linear shapes make reduced precision a candidate;
  actual speed and numerical behavior still require measurements.

NUMA placement and asynchronous checkpoint I/O remain follow-up candidates if
the profiler and whole-run timing justify them. New APIs appearing in current
documentation are not assumed available in the installed PyTorch2.4environment.

## Future training submissions

The four submit wrappers now use `submit_partition_job.py` to create exactly
one `--partition=gpu,gpux` job. The per-run lock and queue-file format are retained;
the queue contains only that job's ID. Existing frozen snapshots keep their
previous submission behavior. Run `pytest -q --capture=sys
tests/test_slurm_submission.py` to check single submission, partition limits,
requeue, dependencies and continuation paths with mocked scheduler commands.
