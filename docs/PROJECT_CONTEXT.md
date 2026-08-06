# AdaViP Project Context

Last updated: 2026-08-03

This document records durable project decisions and interfaces. Update it when
the experiment scope, data contract, architecture, or dependency pins change.
Use `docs/WORKLOG.md` for chronological activity.

## Goal

AdaViP is a task-conditioned adaptive visuotactile perception module for
contact-rich manipulation. Frozen base encoders produce visual, tactile, and
proprioceptive features. A hypernetwork uses the task condition and current
base features to generate modality-specific and fusion transformations. The
adapted representation is consumed by an otherwise unchanged policy backbone.

The target real-robot tasks are:

1. Peeling
2. Wiping
3. Occluded Object Picking
4. Threaded Assembly
5. Puzzle Insertion

## Experiment Scope

Table I contains five methods:

1. VT-DP
2. RDP
3. Policy Consensus
4. VT-DP + AdaViP
5. RDP + AdaViP

ReTac-ACT is not part of the experimental comparison. It may remain cited in
Related Work, but it must be removed from Table I and the corresponding method
list in Experimental Setup.

## Why These Baselines

The three baseline families are complementary rather than redundant:

- VT-DP is the static visuotactile Diffusion Policy reference.
- RDP tests a slow-fast policy with a high-rate tactile feedback path.
- Policy Consensus tests downstream adaptive/compositional fusion of RGB and
  tactile policies.

Together they distinguish adaptive perception from static perception, temporal
reactivity, and adaptation that happens only after modality-specific features
or actions have already been computed.

## VLA Scope

Do not add a VLA such as pi0/openpi to the primary Table I for now. A standard
VLA is normally vision-language-action and does not accept the same GelSight
and proprioceptive inputs. Comparing it directly would confound tactile access,
model scale, language conditioning, and policy backbone, rather than testing
AdaViP's adaptive-perception claim.

A VLA can be a separate future extension only if it is given the same sensory
streams, task specification, action convention, data split, and evaluation
protocol. It should then have its own comparison table instead of replacing a
core visuotactile baseline.

## Staged Validation Plan

Separate hardware validation from policy validation:

0. Use the independent LeRobot collection pipeline to record and replay short
   episodes. Do not run a learned policy yet. Verify camera frames, robot state,
   actions, timestamps, episode boundaries, and safe stop behavior.
1. Before GelSight arrives, train a small RGB + proprioception Diffusion Policy
   smoke test on one low-risk task. Use the official Diffusion Policy reference
   as the first learned method because it has fewer tactile and slow-fast
   dependencies.
2. After GelSight integration, run the static VT-DP path on the same task and
   the same LeRobot-to-canonical adapter.
3. Run RDP with the high-rate tactile view, then Policy Consensus.
4. Add AdaViP to VT-DP and RDP only after the three baseline paths pass offline
   smoke tests and safe real-robot rollouts.

Wiping is a reasonable first task if the planned workspace and safety limits
make it low risk; otherwise use the simplest repeatable task available. Avoid
starting with Threaded Assembly or Puzzle Insertion.

## Data Boundary

Data collection is owned by a separate pipeline. AdaViP and all baselines will
consume an exported LeRobot dataset through project-owned adapters.

The draft common sample contract is:

- wrist RGB image
- third-person RGB image
- GelSight tactile image or tactile frame sequence
- robot proprioception
- action
- task identifier
- timestamp, episode index, and frame index

Record/export data at the highest frequency needed by any method. The current
RDP reference setup records at 24 Hz, trains DP at 12 Hz, and trains RDP at
24 Hz. Dataset adapters should construct the required temporal view without
discarding the high-rate tactile signal in the source dataset.

The exact feature names, state vector, action convention, horizons, and split
policy are not final yet.

## Baseline Sources

The repositories under `third_party/` are reference dependencies. Keep their
working trees unmodified when possible and implement integration in AdaViP.

| Component | Official repository | Pinned HEAD | Role |
| --- | --- | --- | --- |
| Diffusion Policy | `real-stanford/diffusion_policy` | `5ba07ac6661db573af695b419a7947ecb704690f` | Canonical DP reference |
| Reactive Diffusion Policy | `xiaoxiaoxh/reactive_diffusion_policy` | `824c5e8de1fd1811106907a04b5f0186e0138c0b` | Main VT-DP/RDP engineering reference |
| Policy Consensus | `policyconsensus/policyconsensus` | `99638602747433fd47be9cc93e43988d848338da` | Independent comparison baseline |

The RDP repository includes a complete DP training path: `train_dp.sh`, a DP
policy, a training workspace, Hydra configurations, and a visuotactile Zarr
dataset implementation. The canonical Diffusion Policy repository is retained
to audit upstream behavior and avoid treating the RDP fork as an unmodified DP
baseline.

## Repository Boundaries

- `docs/`: paper, figures, durable decisions, and work log
- `third_party/`: pinned upstream baseline repositories
- future `adavip/data/`: stable data contracts and dataset metadata only; do
  not bind this layer to a provisional collection format
- future `adavip/processing/`: format-specific readers, synchronization,
  conversion, temporal sampling, and policy-facing adapters
- `adavip/model/`: project-owned encoders, adaptive transforms, and
  fusion modules
- `adavip/policy/`: AdaViP policy composition around an unchanged
  action-policy backbone
- future `adavip/baselines/`: wrappers for the pinned upstream baselines
- future `adavip/workspace/` and `adavip/env_runner/`: training and rollout
  integration
- future `configs/`: experiment and dataset configurations
- `tests/`: unit and smoke tests

Do not put the external data-collection pipeline in this repository. Do not
commit robot datasets, checkpoints, or generated experiment outputs.

## Open Decisions

- Final LeRobot feature schema and tactile sequence representation
- Robot state fields and action coordinate convention
- Observation/action horizons and control frequencies per method
- Task embedding representation
- Frozen encoder architectures and pretrained weights
- Size and parameterization of generated adaptive encoders
- Shared train/validation/test split and real-robot trial protocol
- Environment isolation strategy for baseline dependency conflicts

## Next Milestone

Keep the AdaViP policy core independent of the provisional collection format,
then audit the input/output contracts of VT-DP, RDP, and Policy Consensus.
Freeze the shared data contract only after the collection pipeline and tactile
stream are concrete. Baseline smoke-training and rollout validation still
precede training or evaluating AdaViP on the robot.
