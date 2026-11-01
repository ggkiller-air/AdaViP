# L1 dropout and paired mask diagnostics

The independent L1 dropout experiment trains from seed42 random initialization,
with no Future or Utility objective. It preserves all original L1 parameters
and full-input behavior, all400 demonstrations/71,265 anchors, task-balanced
sampling, T=2, AdamW LR8e-5, warmup500 and cosine decay. Batch32,100epochs;
fixed checkpoints E20/E40/E60/E80/E100, atomic rolling latest and exact resume.
The runtime matches the latest original L1 profile: FP32, cuDNN TF32 enabled,
cuDNN deterministic=false, matmul TF32 disabled, no autotune.

The November1 rerun restores the original L1 shared protocol: batch64,
500epochs, fixed E50/E100/E200/E300/E400/E500 snapshots and the same500-update
warmup followed by a557000-update total cosine schedule. The defaults and
training entry points now enforce these restored settings. The October31
batch32/E100 artifacts and their frozen source remain the historical experiment
described above. `submit_l1_restored.py` compares all five new launch configs
against the actual original L1 config before submitting any job.

Input sampling is90% full and10% uniformly divided among seven individual
removals (V1,V2,V3,T1,T2,P,L) and two group removals (vision,tactile). Thus10%
is the probability that a sample uses a removal condition, not a per-token
independent probability. Each removal applies to the whole two-frame history.
Removed sensor tokens are zeroed after temporal projection and sensor identity;
six presence flags enter the existing fusion MLP. Removed language is zeroed
before the existing late language head. No learned missing parameters or
language presence dimension are added. No neural-network nn.Dropout is used.
The mask generator is independent of initialization, sampler, and action noise;
its state is saved with model, EMA, AdamW, noise, cursor, and epoch statistics.

A bounded real-data batch32 preflight runs inside the new allocation before
formal training. It checks all ten conditions, full-input exact fusion parity,
finite losses/gradients, and nonzero gradients to all five online ResNets without
updating weights/optimizer/EMA. The first formal update is saved before loading
the279GiB shared RAM cache. Logs include actual condition counts.

The separate offline diagnostic uses the original L1 E500 EMA checkpoint,
all71,265 anchors, ten conditions and four fixed noise/timestep draws per anchor.
No trained policy weights or saved normalizer are changed. One current-observation
encoding is cached per batch; every mask receives the identical noisy action,
target, diffusion timestep and valid-action mask. The full-input adapted path is
checked against the native checkpoint encoder. The diagnostic uses deterministic
cuDNN with TF32, to make repeated model/mask comparisons reproducible.

Outputs are losses.npy [N,4,10], timesteps.npy [N,4], metadata.npy [N,8],
signed_deltas.npy [N,9], protocol.json, report.json and report.md. Metadata records
sample/task/episode/timestep/source indices and complete observation/action flags.
Signed delta is masked loss minus full loss and is never clipped. Reports include
per-task/global/macro losses, positive/zero fractions, signed quantiles, repeat
sign consistency and a separate complete-window summary. Arrays and a strict
continuation contract allow exact restart after scheduler interruption.

This is training-data offline mask sensitivity, not held-out or simulator
success. For no-dropout L1, masked conditions are out of the training distribution;
large or negative effects need inspection before interpreting utility. The
same diagnostic supports L0 and L1-dropout later. L0 retains its native learned
missing-token representation; L1 uses zero post-projection tokens, as recorded
in the protocol. Compare matched dataset/noise/weights/epochs and report normalizer
and representation differences. E500 L1 versus E100 dropout is not an isolated
dropout ablation; the existing original L1 E100 snapshot supports an epoch-matched
comparison when the new run completes.

Submission uses one requeue-enabled H200 job per request, gpu,gpux, the queried
common maximum time,32CPUs/384GiB, explicit srun CPU allocation, isolated output
locks, GPU/process telemetry and frozen source. Existing runs remain untouched.
