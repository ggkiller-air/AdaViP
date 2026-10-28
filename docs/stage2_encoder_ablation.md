# UViP ManiFeel Stage 2 Encoder Training Strategy Ablation

All four strategies use the existing T=2 E500 EMA Teacher and complete weighted
K64 archive: 65,265 windows, nine tasks, 400 demonstrations. S0 is the archived
run; it is not retrained. L1/L2 code and jobs are independent of this experiment.

| Strategy | Context used for z_t | Action ResNets | Action temporal projections |
| --- | --- | --- | --- |
| S1 | Shared E500 observation path | Frozen throughout | E500 init; train from C, LR8e-5 |
| S2 | Shared E500 observation path | E500 init; train from C, LR1e-5 | Same as S1 |
| S3 | Complete frozen E500 observation path | Independent E500 copy; train from B, LR8e-5 | Independent E500 copy; train from B, LR8e-5 |
| S4 | Complete frozen E500 observation path | Independent random init; train from B, LR8e-5 | Independent random init; train from B, LR8e-5 |

A is epoch1 (Utility Predictor only); B is epochs2--6; C is epochs7--600.
S3/S4 Action P/L projections and Sensor Identity are independent E500 copies,
trained from B. Their unused old fusion/missing-token parameters stay frozen.
The complete context encoder, including its P/L, identity and fusion, never
updates. Random S4 initialization preserves the common seed42 RNG stream.
All five Action ResNets are independent parameters, including within a branch.

The original Stage2 Utility Predictor, five identity-initialized HyperNet-FiLMs,
Utility-conditioned Query, four-head Attention Fusion and fresh Diffusion U-Net
are retained. Utility input z_t and Action input u_hat keep their stop-gradients.
Observations are normalized once, then supplied to both branches.

The shared loop retains batch32/600epochs/seed42, S0 task-balanced replacement
sampling with epoch seed42+epoch, complete action masks, RMS-normalized nine-axis
SmoothL1 Utility Loss (beta1), Action Loss +0.05 Utility Loss in B/C, global
gradient clipping1, AdamW betas(0.95,0.999)/eps1e-8/decay1e-6, Action/Utility
LR1e-4, 500-update shared warmup/cosine schedule (floor0.1), EMA0.9999.
The 500-update encoder unfreeze ramp starts at C for S1/S2 and B for S3/S4.

`scripts/manifeel/train_stage2_encoder_ablation.py` reuses the original policy,
data source, optimizer/scheduler and checkpoint helpers. One ablation workspace
serves all four strategies; the old S0 entrypoint remains available.

Before loading image arrays, validate K64 file hashes, Teacher normalizer,
cache source contract/array metadata and DataLoader IPC. On the first actual
raw batch, bounded B/C backward checks verify gradient routing and restore RNG
without optimizer updates. Then perform the next formal A/B/C update, atomically
save latest, and load/hash all 299,494,742,868 decoded bytes with eight threads.
Fork eight persistent workers only after RAM is ready. They share read-only
arrays, use separate core affinity, and supply pinned batches for nonblocking
H2D copies. The next sampler draw follows the saved cursor without repetition.
RAM loading/checkpoint startup time is separate from training throughput.

Per-run files:

- `contract.json`, `architecture.json`, `initialization.json`: strategy, source,
  data and initialization identities; complete inference architecture.
- `metrics.jsonl`: count-weighted Overall/Per-task Action and Utility losses,
  task counts, LR, throughput, loader wait and allocated/reserved CUDA peaks.
  A-stage Action Loss is null because no Action loss is trained in A.
- `gradients.jsonl`: phase parameter counts and backward participation/norms,
  including each ResNet, Action tokenizers, Utility Predictor and U-Net.
- `runtime.jsonl`, `gpu_monitor_JOBID.log`: RAM load time/bytes and GPU telemetry.
- `checkpoints/epoch{1,6,10,20,50,100,200,300,400,500}.ckpt`, rolling `latest.ckpt`:
  raw/EMA, immutable branch, normalizer/RMS, architecture, optimizer/scheduler,
  RNG, metrics and next-batch cursor. Final latest contains completed E600.

Frozen raw and EMA weights are hashed against their initial E500 values before
every checkpoint. The portable evaluation loader restores all strategies
strictly without Teacher, Utility labels or training data. Rollouts report the
official `test/mean_score` as per-task SR and its nine-task macro mean as Avg SR.

Submit each strategy with isolated output/source paths:

```bash
S2A_SOURCE=/data/wangzihao/outputs/manifeel/RUN/source \
S2A_OUTPUT=/data/wangzihao/outputs/manifeel/RUN/s3 \
S2A_STRATEGY=s3 bash scripts/manifeel/submit_stage2_encoder_ablation.sh
```

The helper queries both current partition maximum walltimes, queues gpu and
gpux, and uses a per-run lock plus sibling-only cancellation. Safe stops resume
in a subsequent allocation from the latest complete optimizer update.
