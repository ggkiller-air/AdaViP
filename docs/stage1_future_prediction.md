# Stage 1 future supervision on L1

The unchanged L1 policy compresses three visual streams, two TacRGB streams and
proprioception into a 256D `z_t`. The action condition is still
`Linear(768,256) -> SiLU -> Linear(256,256)` on `[z_t, language512]`.
Five separate ResNet18/GroupNorm backbones are randomly initialized and trained
with the existing policy's AdamW optimizer and learning-rate schedule. Original
L1 supervises only action denoising; there is no Utility or Future objective.

New modules implement training-only future branches without changing the policy
class, inference path, existing trainers or Stage 2. All groups have identical
policy initialization, verified by a SHA256 over every policy parameter/buffer.
Head construction has its own isolated per-sensor RNG stream. No trained L1 or
Stage 2 checkpoint is used to initialize any experiment or target encoder.

| Group | Targets | Space | Formal training |
| --- | --- | --- | --- |
| A | None | Existing baseline | No training or GPU sanity |
| B | Left/right TacRGB | Raw pixels | Yes |
| C | Left/right TacRGB | EMA ResNet features | Yes |
| D | Front/side/wrist + left/right TacRGB | Raw pixels | Yes |
| E | Front/side/wrist + left/right TacRGB | EMA ResNet features | Yes |

Each independent head receives only pre-language `z_t` and predicts a single
frame at **t+4**. Raw heads use independent MLP-to-4x4 plus upsample/conv/GroupNorm
decoders, producing 64x64 RGB in [0,1] with sigmoid outputs. Raw targets are
bilinearly resized with antialiasing; loss is pixel MSE. Latent heads are
256->512->512 MLPs with SiLU and cosine loss on normalized features. Target
ResNets are initialized as independent copies of their online counterparts;
they stay in eval mode, accept no gradients, and update after each optimizer
step with fixed momentum 0.996. Floating buffers also follow EMA, while integer
counters are copied. This target EMA is separate from the original policy EMA,
whose existing warmup/decay schedule remains unchanged.

Each sensor loss is averaged over valid future samples. T averages two tactile
losses; V averages three visual losses; V+T averages V and T equally. Invalid
episode-end targets are zero/masked and are never encoded. Original action
supervision remains active for all anchors, including those without a valid
future. Future features/images never enter current observations or inference.

An independent eight-batch calibration uses the ratio of gradient L2 norms over
all `obs_encoder.fusion` parameters. The fixed future weight is the median of
`0.1 * ||grad(action)|| / ||grad(future)||`. Calibration performs no optimizer
or EMA updates and leaves policy parameters and training RNG unchanged.
Each experiment's final weight and per-batch ratios are saved separately.

Formal runs use batch32, 100 epochs, seed42, AdamW LR8e-5, 500-update warmup
and cosine decay. Sampling, observation preprocessing, all 400 demonstrations /
71,265 anchors and the lack of a validation split follow the original L1 run.
There is no additional observation augmentation. Fixed checkpoints are E20,
E40, E60, E80, E100; rolling latest also preserves optimizer, policy EMA,
auxiliary heads, target EMA, sampler cursor and RNG for exact continuation.
Fixed checkpoints are never overwritten. All groups log action/future/total
loss, every sensor loss, valid target counts and periodic shared-gradient norms.

`sanity_stage1_future.py` exercises the four new full-size policies with real batch32
data on one allocated H200, checks worker affinity/IPC separately, calibrates
weights, verifies online/target updates and saves checkpoints. Formal submission
requires its PASS marker and matching source fingerprint. `submit_stage1_future.py`
creates only B--E, each with a new config/output directory and one Slurm job
eligible for both gpu/gpux. Submission queries both limits and chooses the common
maximum; per-run locks, requeue and safe checkpoint continuation are enabled.

This experiment performs neither Stage 2 training nor rollout evaluation.
