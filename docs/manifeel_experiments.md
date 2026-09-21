# ManiFeel experiment entrypoints

The active ManiFeel entrypoint is the single-task Diffusion Policy path:

```bash
bash scripts/manifeel/smoke_train_dp.sh
bash scripts/manifeel/train_single_task_dp.sh
```

The default task is USB with `vision_wrist`. The task, Isaac Gym config, data
root, and output root are controlled by `MANIFEEL_TASK_CONFIG`,
`MANIFEEL_ISAACGYM_CONFIG`, `MANIFEEL_DATASET_PATH`, and the variables in
`scripts/manifeel/common.sh`.

Only `train_multitask_diffusion_workspace.yaml` remains as a historical
nine-task training example. Older FM/AdaViP experiments are not new training
entrypoints; their complete resolved configs are stored in their checkpoints,
and checkpoint evaluation tools remain available for historical analysis.

## Power Plug Insertion: upstream single-task baselines

The profiles `configs/manifeel/power_plug_{vision,tacrgb,tacff}.yaml` use the
unchanged upstream `train_diffusion_workspace.yaml` and the respective
`vision_wrist`, `vistac_wrist`, and `visff_wrist` task configs. All three use
`plug_quan_Aug02`, 50 demonstrations, seed 42, and 1,000 epochs. All three
use batch size 512, eight persistent training workers, prefetch factor one,
and the same validation settings. Each policy gets a separate output
directory. Periodic simulator rollouts are disabled because they are not
needed to train the three policies. Validation runs every 10 epochs; periodic
checkpoints are saved every 100 epochs and all retained, including a final
`latest_epoch1000.ckpt`. A separate best-validation checkpoint is also kept.
Evaluate the three policies under one fixed rollout protocol after training.
This upstream TacFF route
encodes the force field through the standard DP image encoder.

Check the profiles without a GPU:

```bash
for method in vision tacrgb tacff; do
  /public/home/wangzihao/.local/miniforge3/envs/manifeel/bin/python \
    scripts/manifeel/train_power_plug_baseline.py "$method" --dry-run
done
```

The current jobs were submitted as vision `308521`, TacRGB `308522`, and TacFF
`308523`. For a later run, inspect the GPU allocation and perform a bounded
preflight before submitting each method with
`sbatch slurm/manifeel/train_power_plug_baseline.sbatch METHOD`. The batch
file requests one scheduler-selected GPU and never pins a node.

Generated Python caches, package metadata, and test caches are intentionally
removed from the working tree. Dataset stores, checkpoints, logs, and videos
remain outside the repository under the configured `/data` roots.
