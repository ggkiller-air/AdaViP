# Single-task DP architecture review

- Rechecked pinned ManiFeel `7d117b79f19e7d0cd2e7529ac8082a544f2839b9`
  and Diffusion Policy `5ba07ac6661db573af695b419a7947ecb704690f`.
  `train_diffusion_workspace.yaml` uses independent, randomly initialized
  ResNet18/GroupNorm encoders, direct feature concatenation, two observation
  frames as global conditioning, H16/Ta8, and 100-step epsilon DDPM with
  ConditionalUnet1D widths [512,1024,2048]. Train/validation batch8,
  AdamW8e-5, 500-update warmup, cosine schedule and EMA are configured.
- `vistac_wrist` uses right TacRGB + wrist + state: per-frame dimensions
  512+512+7=1031, flattened two-frame conditioning 2062. The U-Net already
  contains FiLM conditioning; our sensor-side Utility/HyperNet/fusion modules
  are additional mechanisms. Single-task configs have no language input.
- A single-task adaptation of the current method should distinguish the
  original modality baseline from a baseline using the same expanded sensors.
  A task-specific Teacher/normalizer/paired Utility labels are needed for a
  clean independent experiment; sensor removals and Utility dimensions must
  follow the chosen inputs. This turn reviews architecture only; no training,
  evaluation, existing process or implementation was changed.

## Single-task Teacher direction

- User selected five sensor streams and a newly trained dropout Teacher, with
  simulator SR checked against the official task result before proceeding.
  Task selection is pending; Power Plug was suggested because batch8 baselines
  and evaluation records already exist. No new workload has been launched.
- Proposed inputs: front/side/wrist + left/right TacRGB, plus retained robot
  state; omit language. A small state projection (T2 x 7 -> 256) makes six
  tokens with the five image-history projections. Teacher fusion can remain
  Concat+MLP; a state-token projection does not require a compression Transformer.
- Proposed first dropout protocol: 80% full, 20% split across five individual
  image removals and two group removals (vision/tactile), preserving state.
  This gives seven Utility targets; including a state-removal intervention would
  instead give eight. These are proposals, not implemented/approved settings.
- Compare matched five-sensor full-input DP with and without dropout, then
  compare full-input EMA rollout SR with the official result under matched
  demonstration and evaluation settings. More inputs and dropout do not
  guarantee higher full-input SR; loss alone cannot qualify the Teacher.
