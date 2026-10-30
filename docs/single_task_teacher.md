# Single-task DP Teachers: batch8 / 400 epochs

Twelve independent seed42 runs:

| Task | Image inputs | Full / removal probability |
| --- | --- | --- |
| Power Plug | front, side, wrist, left TacRGB, right TacRGB | 100/0 and 60/40 |
| USB Insertion | same five inputs | 100/0 and 60/40 |
| Ball Sorting | same five inputs | 100/0 and 60/40 |
| Power Plug | wrist, right TacRGB | 100/0 and 60/40 |
| USB Insertion | wrist, right TacRGB | 100/0 and 60/40 |
| Ball Sorting | front, right TacRGB | 100/0 and 60/40 |

Every run includes normalized robot state, with no language. Each independent
randomly initialized ResNet18/GroupNorm outputs 512D per frame; each two-frame
history is projected with Linear(1024,256), LayerNorm, SiLU. State uses
Linear(14,256), LayerNorm, SiLU. Learned sensor identities are added. Concat of
image/state tokens and image-presence flags feeds a 256D MLP global condition.
No Utility, HyperNet, attention or future-prediction modules are included.

Five-input dropout draws full with probability0.6; each of five individual
image removals, all-camera removal and both-tactile removal has probability
0.4/7. Two-input dropout uses full0.6, camera removal0.2, tactile removal0.2.
Exactly one condition is drawn per sample and shared across both history
frames. Robot state remains present. Removed independent tokens are replaced
before fusion with learned missing tokens, with explicit presence flags.

The pinned official settings retained here are train/validation batch8,
AdamW(lr8e-5, betas0.95/0.999, eps1e-8, weight_decay1e-6), 500-update
warmup/cosine, FP32, EMA(power0.75, max0.9999), shuffled nonreplacement epochs,
H16/To2/Ta8, 100-step cosine epsilon DDPM and the original 1D U-Net widths
[512,1024,2048]. No gradient clipping or AMP is introduced. Image sizes are
256x256 for both camera and tactile inputs; image range normalization then
ImageNet transform follows the pinned official pipeline. Action supervision
uses the official padded sequence windows and native task action dimensions
(Power Plug/USB6, Ball Sorting7), without multitask action padding or masking.

The requested 400-epoch budget replaces the official 50-demo launch example's
1000 epochs. The new token/fusion architecture and additional inputs are
intentional experimental changes. The two-input versions include the requested
state tokenizer and therefore are not verbatim replicas of the official model.
Train-only state/action normalization avoids fitting held-out trajectories;
the pinned official loader fits these statistics on the entire store instead.

Each local source store contains50 episodes: the official2% episode split
produces49 training episodes and1 validation episode, using seed42. Both
dropout variants share split, normalizer, initialization and epoch sample order.
Condition sampling uses an independent saved generator, so it cannot perturb
action noise. Independent runs use the same seed to isolate dropout effects;
three tasks do not replace repeated training seeds.

The trainer retains full Hydra config, raw/EMA weights, frozen normalizer,
Adam/scheduler, EMA counter and RNG/batch cursor. Latest saves after the first
update and every1000 updates; fixed snapshots are saved at50/100/.../400.
EMA paired validation (same noise/timesteps for all masks) runs every10 epochs
and saves best full-input validation loss. Validation never selects on raw
loss for an EMA model. Simulator rollouts are separate: use50 environments,
500 steps and `test/mean_score`, with independent checkpoint-selection and
final-test seeds when comparing with paper SR. No SR is inferred from loss.

Launch one eligible `gpu,gpux` job per run via `submit_partition_job.py`.
Partition limits are queried at each submission, using their common maximum.
Each job has one H200,16 CPUs,128GiB, requeue eligibility, a per-run lock,
explicit srun CPU count, and persistent output/cache paths. A separate40-update
GPU preflight checks actual throughput, loader wait, all requested CPU/worker
affinities, checkpoint persistence and strict EMA/normalizer inference restore.
Production starts from the original seed after preflight and reuses decoded
read-only source arrays. Runtime diagnostics do not change scientific settings.
Scheduler stop files trigger an atomic checkpoint and a dependent continuation;
application errors fail instead of automatically retrying.

Example on approved training resources, after sourcing `common.sh` and
`activate_manifeel`:

```bash
python scripts/manifeel/train_single_task_teacher.py \
  --task power_plug_insertion --sensors five --dropout \
  --output /data/wangzihao/outputs/manifeel/EXPERIMENT/power_plug_five_drop40_batch8_ep400_seed42
```

Use `--check-inputs` for dataset/IPC validation without constructing a model or
requiring a GPU. Omit `--dropout` for the matched no-dropout version. Checkpoints
contain a device-aware runner config and can use `eval_single_task_checkpoint.py`;
formal reports must use its `test/mean_score` field.
