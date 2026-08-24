# Piper RDP at 18.75 Hz

These local Hydra task and AT profiles integrate the Piper pick-and-place
dataset without modifying the pinned RDP source. The policy uses two RGB views,
7D absolute joint state, 15D task-local GelSight PCA features, and 7D absolute
joint/gripper actions. Full-rate tactile embeddings condition the AT decoder.

Prepare data with `scripts/real_world/prepare_piper_rdp_dataset.py`, then use
`scripts/real_world/train_piper_rdp.py`. GPU training must follow the repository
interactive preflight and Slurm submission rules.
