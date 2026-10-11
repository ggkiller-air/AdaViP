# Modulation failure-type audit (2026-10-11)

- User requested failure-type recheck of existing success/failure swaps and a
  two-task training choice with6D initially-held vs7D later-grasp behavior.
  Read recent mechanism and cross-task logs; no new GPU work or training.
- Preselect descriptive failure geometry: no target contact; contacted but
  never near; approached near then departed; near but unfinished. Near=2x
  pinned task keypoint-success threshold; sensitivity1.5/2/3/4x. Do not infer
  missed grasp/drop/jam from geometry without grasp or finger-contact traces.
- Reuse normal/zero pairs for Power Plug fusion/no-fusion, Peg Insertion and
  Nut-Bolt, two evaluation seeds each. Report helpful normal-only vs harmful
  zero-only outcomes using the failed counterfactual's category; groups are
  post-intervention descriptive outcomes, not pre-treatment causal subgroups.
- Task-code audit: current nut-bolt and bulb reset code places the object in
  hand and closes the gripper before policy starts.7D action alone does not
  mean later grasp. Object Search does not close gripper during reset; any
  future use must correct its known wrist/front observation mismatch and
  retrain/evaluate all compared models with the same view. No local training.
- Completed CPU report under `results/modulation_failure_types_20261011/`;
  primary counts (no contact/contacted far/near then departed/near unfinished),
  normal-only vs zero-only: Power fusion 0/1/0/1 vs 0/4/0/0; Power no-fusion
  2/6/0/4 vs 0/3/0/3; Peg 0/6/0/1 vs 0/4/0/1; Nut 0/0/4/7 vs 0/0/4/9.
  Same coarse failure class appears in both directions, not a stable rescue
  class. Nut swaps all approached near across1.5–4x; Peg far/near split is
  threshold-sensitive. Power no-fusion counts stable across all thresholds.
- Paired initial state/poses match elementwise;500x50 reward maxima match
  per-env logged scores and mean_score. Success-time geometry also matches
  each task's pinned success threshold. Figure visually checked; README
  records post-treatment grouping, missing grasp evidence, and small samples.
- Recommend Power Plug6D plus corrected-front Object Search7D, with actual
  demo grasp timing checked before training. Nut/Bulb7D are initially-held.
  Proposed static FiLM means input-independent learnable gamma/beta; retain
  same ResNet/fusion/modulation form and batch8. This tests conditioning, not
  a parameter-matched generator architecture. Train only on engram.
- Focused failure-type/mechanism tests:17 passed in1.65s. Python3.8 initially
  rejected built-in generic test annotations; fixed with future annotations.
- Further task-selection audit supersedes the initial Object Search choice:
  its success predicate only checks a0.12–0.20m keypoint-distance band, not
  grasp. Prefer verifying Ball Sorting as the7D candidate: reset places its
  manipulated object near the base (`before_move_to_grasp=True`), not in-hand;
  success checks mean vertical keypoint separation >0.1m. Reset still closes
  the gripper, so actual initial grasp state/demo timing needs verification
  before claiming a later-grasp contrast. Correct front view in all arms.
- Full suite:127 passed,1 skipped,3 failed in13.14s; unchanged failures are
  two task-profile checks missing pih_quan_June06 and one ball-sorting launcher
  dry-run missing sorting_quan_Aug8. No download or training to repair these.
  New source/tests py_compile and git diff --check passed.
