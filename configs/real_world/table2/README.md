# Table 2 offline training

This directory owns the server-side training protocol for the real-robot
experiments. Robot control, camera processes, sensor publishers, teleoperation,
and online rollout code are intentionally out of scope.

`third_party/reactive_diffusion_policy` is the shared upstream implementation
for both baselines:

- `vtdp`: the upstream visuotactile Diffusion Policy workspace;
- `rdp`: the upstream two-stage Asymmetric Tokenizer (AT) and Latent Diffusion
  Policy (LDP) workspaces.

The repository does not modify the pinned submodule. Task/config selection is
recorded in `manifest.json`, and `scripts/real_world/train_table2.py` invokes
the pinned upstream training entrypoint with explicit dataset and output paths.

## Protocol status

The upstream repository contains reference configurations for Peeling and
Wiping. They are useful for validating the offline training path, but they are
not yet formal Table 2 configurations: their observation schemas do not yet
match the paper's complete two-view, GelSight, and proprioception protocol.
The other three tasks need configs derived from the collected Zarr schemas.

Consequently, reference runs require `--allow-reference-config`. Formal runs
must remain blocked until each manifest entry has `protocol_status` set to
`formal_ready` after its observation/action keys and sampling frequency are
verified.

DP and RDP use different processed datasets:

- VT-DP consumes a 12 Hz store, normally produced by downsampling 24 Hz data
  by a factor of two.
- RDP AT and LDP consume the corresponding 24 Hz store.

Never point both methods at the same store unless its sampling semantics have
been checked explicitly.

## Commands

Inspect a command without requiring a GPU or environment:

```bash
python scripts/real_world/train_table2.py \
  --method vtdp \
  --task peeling \
  --dataset /data/wangzihao/datasets/real_world/peeling_12hz.zarr \
  --allow-reference-config \
  --dry-run
```

Run through the environment wrapper:

```bash
TABLE2_METHOD=vtdp \
TABLE2_TASK=peeling \
TABLE2_DATASET_PATH=/data/wangzihao/datasets/real_world/peeling_12hz.zarr \
TABLE2_ALLOW_REFERENCE_CONFIG=1 \
bash scripts/real_world/train_table2.sh
```

For RDP, `--stage all` trains AT first and passes its `latest.ckpt` to LDP.
Use `--stage at` or `--stage ldp --at-checkpoint PATH` to manage the stages
independently.
