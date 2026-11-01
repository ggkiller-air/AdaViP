# OpenViTac data and UViP integration

This benchmark lives in an independent worktree/branch, with all new Python code
under adavip/openvitac. ManiFeel and UniVTAC modules, data, normalizers, labels,
checkpoints and running jobs are not modified.

## Source and action contract

The pinned Hugging Face revision is
5fb2757b88826917242adefd9196424ef8cf2d98. Download only
Simulation/GelSight_mini: 6,599 files, 9,455,749,868 bytes, eleven benchmark
tasks, 1,300 episodes and 306,218 recorded frames. insert_block_v2 has 300
episodes; the other tasks have 100 each. Every downloaded file must match its
published Git/LFS object hash. The domestic mirror is only a transport route.

Primary sources:

- [Simulation dataset README](https://huggingface.co/datasets/FVL-Lab/OpenViTac/blob/5fb2757b88826917242adefd9196424ef8cf2d98/Simulation/GelSight_mini/insert_USB/README.md)
- [Simulation metadata](https://huggingface.co/datasets/FVL-Lab/OpenViTac/blob/5fb2757b88826917242adefd9196424ef8cf2d98/Simulation/GelSight_mini/insert_USB/meta/dataset_metadata.json)
- [Official next-state converter](https://github.com/FVL-Repo/OpenViTac/blob/5220aaf6a68dec5d2a6d284a76cffe337ef234c9/policy/openpi/abs_joint/convert_insert_usb_sim_to_lerobot.py)
- [Official control implementation](https://github.com/FVL-Repo/OpenViTac/blob/5220aaf6a68dec5d2a6d284a76cffe337ef234c9/envs/robot/robot.py)

**The public simulation release contains no action column.** Its eight state
values are measured arm joints (seven radians) and gripper open fraction.
Those are not eight recorded joint-position commands. This adapter explicitly
constructs a proxy target at row t from measured q0..q6 at row t+1 and the
**left finger position in metres** at row t+1. The open fraction is never
passed to the simulator as finger metres. This agrees with the official
abs_joint next-state preprocessing convention, but does not recover original
position/velocity/force targets.

The codec name is derived_next_observed_abs_joint_left_finger_v1. It is saved
in every dataset/label manifest. The final observation row has no target. An
action crossing a non-nominal source step gap is also invalid and set to zero;
it is excluded from windows and normalization. Normalization is pooled over
training episodes, not task-specific, and remains frozen downstream.

## Time, modalities and instructions

Images are head/wrist RGB at H270 x W480 and marker-bearing tactile_left/right
RGB at H240 x W320. MP4 frame rate and published timestamps are nominally 60 Hz;
timestamp equals frame_index / 60. Underlying physics is nominally 120 Hz with
two steps per saved frame. The release has 201 exceptional edges across 151
episodes: empty_can_selection (100 edges), roughness_guided_regrasp (50), and
grasp_chip (51 terminal one-step edges). Published timestamps hide these gaps.

The converter reconstructs relative source simulation time using the gap
manifest. It retains original timestamps as provenance and splits valid
windows at **all** non-two-step edges. H16 / history2 windows additionally
require the next-state target row: total complete windows are 282,967 before
train/validation partitioning. No windows cross source episodes or gaps.
Future targets use the same segment boundaries. A four-frame future offset is
nominally 66.7 ms, not a ManiFeel timing assumption.

The model receives exactly V1=head, V2=wrist, T1=tactile_left,
T2=tactile_right, P=measured 8D state, L=frozen 512D CLIP embedding.
Four independent ResNet18/GroupNorm branches retain the original temporal
projection, Concat+MLP fusion, 256D conditioning and ConditionalUnet1D DP.
The U-Net widths stay 512/1024/2048; only its action input/output dimension is
eight. Existing ManiFeel Teacher weights/normalizers are not imported.

Language is resolved from each row's local task_index via tasks.jsonl. There
are 13 task/instruction entries and 12 unique strings: insert_block_v1 and the
cube variant of insert_block_v2 share text. All three v2 instructions remain
distinct. Benchmark task identity (eleven tasks) is retained as metadata for
sampling/metrics and never replaces language. Source variants such as hard,
soft, heavy or light are split-stratification provenance, not network inputs.
The frozen model is the existing cached openai/clip-vit-base-patch32; no hash
or random debug embedding is used.

## Converted format and entrypoints

Raw data root: /data/wangzihao/datasets/openvitac/raw/5fb2757b

Converted root: /data/wangzihao/datasets/openvitac/converted/uvip_v1_5fb2757b

Each task stores contiguous, mmap-compatible NPY state/action/validity/language/
timestamp/source-time/EEF/finger arrays, episode ends, complete-window indices
and an episode-to-MP4 map. Images remain in the verified source files. This is
an MP4-backed UViP adapter, not a claim that the ManiFeel Zarr reader accepts
the new source unchanged. The loader returns the existing obs/action/
action_loss_mask batch API plus task, episode, window and language identities.

The default processed camera shape is [3,256,256] with aspect-preserving
letterbox. Tactile stays [3,240,320]; do not transpose it to ManiFeel's shape.
PyAV explicitly decodes RGB, with no second BGR swap. Each worker retains one
decoded uint8 episode in a bounded LRU. Batches convert to [0,1] float32 and
apply the existing image range normalizer exactly once to [-1,1]. No full
float32 dataset cache is created. A future high-throughput cache must preserve
these exact resize/color rules and use a new version if changed.

Run from the independent worktree:

    source scripts/openvitac/common.sh
    "$OPENVITAC_PYTHON" scripts/openvitac/convert.py --raw-root /data/wangzihao/datasets/openvitac/raw/5fb2757b --output /data/wangzihao/datasets/openvitac/converted/uvip_v1_5fb2757b
    "$OPENVITAC_PYTHON" scripts/openvitac/smoke.py --root /data/wangzihao/datasets/openvitac/converted/uvip_v1_5fb2757b --output /data/wangzihao/outputs/openvitac/a_fresh_smoke --full-model --device cpu

The environment is /public/home/wangzihao/.venvs/openvitac: an isolated venv
over the existing PyTorch 2.4.1/Python 3.8 stack, with pyarrow 17 installed only
in the venv. The old environment is not changed. Offline policy/data work
does not require Isaac Sim. The 3090 deployment separately needs the official
Python 3.10 / Isaac Sim 4.5 / Isaac Lab 2.1.1 / bundled modified TacEx stack.

The independent OpenViTacWorkspace retains the original task-balanced sampler,
EMA, scheduler, paired per-task micro/macro validation and atomic checkpoints,
with the new condition constants and dataset interface. Source fingerprints are
part of resume provenance. Its bounded-step unit test saves at step 1, resumes
to step 2, stays at epoch 0, and verifies all nine validation conditions.
scripts/openvitac/train.py refuses an unbounded run unless explicitly enabled.
No real-data epoch run is started during this preparation.

## Stages 0.5, 1 and 2

The declared conditions are full, V1, V2, T1, T2, P, L, vision, tactile:
**nine conditions, eight signed Utility dimensions, six presence flags**.
With full-input probability .8, each removal has probability .025. There is
no dummy third visual input and no zero-padded obsolete Utility dimension.

- Stage 0.5: paired denoising loss reuses the same action, noise and timestep
  across all nine conditions per draw. Raw differences preserve signs.
  The bounded label helper defaults to two IID uniform-timestep draws; it is
  not the previous ManiFeel stratified K64 protocol. Large-scale generation is
  disabled in the config. Formal K64 scheduling/precision must be separately
  frozen before producing labels.
- Stage 1: label joins verify dataset/action/Teacher/normalizer/mask contracts
  and stable window identity. Utility heads output eight targets. Future
  targets are separate from causal observations and have gap/episode validity;
  two tactile or four visual+tactile raw heads are supported. The new raw-head
  adapter does not claim equivalence to every ManiFeel EMA-latent ablation.
- Stage 2: exactly four image HyperNets modulate V,V,T,T. P and L remain
  unmodulated. Signed train-only RMS normalization is provided. Action gradients
  cannot reach the Utility predictor, and Utility gradients cannot reach the
  encoder. A/B/C trainability is explicit. Teacher encoder/normalizer copying
  leaves the independently constructed DP fresh. The implemented final fusion
  is the independently copied Teacher MLP, with identity-initialized FiLM.

## Storage and formal-training gates

The initial format adds only numeric/index/embedding metadata to the 8.81 GiB
source videos. A full uint8 processed image cache at V256 and native tactile
would be 243.55 GiB before compression; float32 would be 974.22 GiB. Do not
carry over ManiFeel's float32 RAM cache strategy to a 384 GiB host allocation.
An all-224 alternative is 171.71 / 686.86 GiB (uint8 / float32), but changes
preprocessing and must have its own protocol version.

For all 282,967 complete windows and candidate A8/H16/K64, saved FP32 Gaussian
noise alone is 8.64 GiB; timesteps and per-draw loss/differences bring detailed
archives to roughly 10-12 GiB. Aggregated float64 Utility/variance/mean losses
plus indices are below 0.1 GiB. Persisted seeds may replace noise only under an
explicit reproducibility contract. Old ManiFeel noise and labels cannot be
reused across action shapes or datasets.

The full four-ResNet DP is approximately 302M parameters. FP32 inference weights
are around 1.1 GiB; raw+EMA+Adam state are around 4.5-5 GiB per resumable
checkpoint. Reserve 6 GiB each, about 50-80 GiB per single-seed stage depending
on snapshot retention, and 200-300 GiB for three stages plus safe temporary
writes. Initial preparation needs about 10 GiB. A practical single-seed formal
storage reservation with the optional uint8 cache is 0.6-0.8 TiB; multiple
seeds/ablations multiply checkpoint storage.

Before full training: validate the derived-target controller on the independent
3090 simulator, especially 60 Hz vs the default 120 Hz take_action call,
gripper units, position-write/PD behavior and fragile contact tasks. Ground-truth
position/velocity/force commands cannot be recovered from the released states.
If next-state replay is inadequate, obtain raw command logs or define and
validate a different codec before treating losses as a benchmark result.
scripts/openvitac/export_replay.py exports small measured/target/EEF/time traces
for the independent 3090 checks. These do not include original object poses,
collection RNG state or original command logs; exact contact replay requires
matching scene initialization and the original collection/control contract.

Also required: select the final split/teacher dropout and Stage 1 design and
exercise the new workspace's real H200 throughput and full-model checkpoint
continuation. The supplied batch8/E500 settings are a starting configuration,
not tuned benchmark results. The old ManiFeel workspace hardcodes condition/
task contracts and must not be pointed at this dataset. No complete epoch or
large Utility generation is part of this preparation task.

## Verified preparation result (2026-11-01)

Download and conversion are complete. All 6,599 source hashes and all 5,200
video headers pass; 1,300 episodes / 306,218 frames produce 282,967 complete
windows, split into 255,040 train and 27,927 validation windows. Episodes are
disjoint: 1,172 train / 128 validation. Converted arrays, indices and language
metadata occupy 47,122,176 bytes in addition to source MP4s.

The real-data CPU smoke covers fourteen windows from all eleven tasks and all
twelve unique instruction strings. A mixed-task batch2 passes forward/action
loss/backward using four full ResNet18 encoders and the original full-width
U-Net. Parameter count is 301,447,174 and random-initialization Action Loss is
1.315205; all image/P/L/denoiser gradient norms are finite and positive. This
is an interface/gradient check, not a trained Teacher result. Stage0.5 evaluates
only two windows with two paired draws, and Stage1/Stage2 backward checks pass.
No real optimizer step or epoch is run. Full report:
/data/wangzihao/outputs/openvitac/cpu_smoke_full_20261101/report.json.

Twenty-one new tests pass, including save/resume at steps1/2 while epoch stays0,
plus59 related legacy compatibility tests. The complete frozen source and the
pinned Diffusion Policy dependency pass the same21 new tests. Frozen source:
/data/wangzihao/outputs/openvitac/source_20261101_v2.

H200 bounded smoke job412580 is submitted with nice10000, a20-minute limit,
eight CPUs/32GiB, and the immutable source/dependency snapshot. Its initial
status is PENDING/AssocGrpGRES; no GPU pass is claimed while pending. Original
training/queued jobs are unchanged. GPU performance/VRAM checks, full-size
optimizer/checkpoint continuation and the3090 controller/contact checks remain
before a formal run.
