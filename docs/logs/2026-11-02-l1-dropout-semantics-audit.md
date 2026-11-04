# Actual L1 dropout sampling and intervention semantics

- Read-only audit of job412407's saved config.yaml, launch_config.yaml, queue
  record and frozen runtime source under
  /data/wangzihao/outputs/manifeel/l1_restored_batch64_ep500_seed42_20261101/.
  Actual workspace=L1DropoutWorkspace; batch64, E500, To2, token_dim256.
- Actual full_input_probability=.9 and mask_weights=[1]*9. Workspace validates
  those exact values. Shared generic helper default=.8 is not used here.
  Each training sample draws one categorical ID with probabilities
  [.9]+[(1-.9)/9]*9 using an independent saved RNG. Workspace passes IDs to
  compute_loss_per_sample, bypassing the policy's fallback sampler; there is
  no second random removal. Nine independent Bernoulli(.1) gates would give
  1-.9**9=.612579511, but that mechanism is not present in this run.
- Condition order: full,V1,V2,V3,T1,T2,P,L,vision,tactile. V1/V2/V3 remove
  front/side/wrist; T1/T2 remove left/right_tactile_camera_taxim; P removes
  state (7D world EE position+xyzw quaternion); L removes task_embedding;
  vision removes all three cameras; tactile removes both TacRGB streams.
  Every removal pattern has unconditional probability1/90 (1.111111%).
  Individual camera/TacRGB marginal removal probability2/90; P,L each1/90.
- One ID per sample for the whole two-frame history; independent across
  samples in the batch, resampled on each training draw. Group removals drop
  multiple streams in one pattern; arbitrary pattern combinations cannot occur.
- Image/TacRGB encoding: per-frame ResNet18 -> concatenate2x512features ->
  Linear/LayerNorm/SiLU256temporal token -> add sensor identity -> mask token
  to exact zeros -> fusion. P similarly projects2x7state values before masking.
  Six sensor presence flags accompany flattened tokens into the fusion MLP.
  L zeros its512D embedding before the late language head; no language presence
  flag or learned missing token. Original observations remain unchanged; all
  encoders still execute. Removal zeros identity as well as projected features.
- At16:48CST, complete epochs1--227 logged16,177,155sample draws: full14,557,976
  (89.99095329%); removal1,619,179 (10.00904671%). Removal counts in order:
  [180376,180690,180045,180180,179347,179513,179687,179761,179580].
  Counts include repeated training draws, not unique dataset anchors.
- Default predict_action has presence=None and no random removal. Eval-mode
  compute_loss defaults to full when condition IDs are omitted; explicitly
  supplied diagnostic masks still work. This workspace forbids held-out
  validation episodes; full-input epoch probes are training-data diagnostics.
- Source locations: adavip/manifeel/l1_dropout.py:16/36/55/63/76;
  full_sensor_schema.py:9/60/70; language_conditioned_obs_encoder.py:61/86;
  language_conditioning_workspace.py:239; reference_diffusion_policy.py:149/188.
