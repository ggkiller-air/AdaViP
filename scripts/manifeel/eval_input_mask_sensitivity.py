"""Export full-dataset paired mask losses with resumable arrays and task summaries."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import time

import hydra
import numpy as np
from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader, Subset

from adavip.manifeel.full_sensor_evaluation import load_evaluation_policy
from adavip.manifeel.full_sensor_schema import CONDITION_NAMES
from adavip.manifeel.full_sensor_workspace import manifest_fingerprint, normalizer_fingerprint
from adavip.manifeel.input_mask_sensitivity import paired_mask_losses, summarize_losses
from adavip.manifeel.language_conditioning_dataset import initialize_worker
from diffusion_policy.common.pytorch_util import dict_apply
from scripts.manifeel.language_runtime import configure_profile


def atomic_json(path: Path, value: dict) -> None:
    """Commit the cursor only after every corresponding output array is flushed."""
    temporary = path.with_suffix(".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def main() -> None:
    """Evaluate one fixed checkpoint without optimization or simulator rollouts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    output = Path(cfg.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    stopped = []
    for signum in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, lambda number, frame: stopped.append(number))
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or "H200" not in torch.cuda.get_device_name(0):
        raise RuntimeError("Diagnostic requires one independently allocated H200")
    flags = configure_profile("tf32_deterministic")
    checkpoint = Path(cfg.checkpoint).resolve()
    policy, model_cfg, metadata = load_evaluation_policy(checkpoint, device="cuda:0", weights=cfg.weights)
    dataset_cfg = OmegaConf.create(OmegaConf.to_container(model_cfg.dataset, resolve=True))
    dataset_cfg._target_ = "adavip.manifeel.language_conditioning_dataset.LanguageConditioningDataset"
    dataset = hydra.utils.instantiate(dataset_cfg)
    if len(dataset) != cfg.expected_samples:
        raise ValueError("Unexpected diagnostic anchor count")
    implementation = hashlib.sha256()
    root = Path(__file__).resolve().parents[2]
    for name in ("adavip/manifeel/input_mask_sensitivity.py", "adavip/manifeel/l1_dropout.py",
                 "adavip/manifeel/language_conditioned_obs_encoder.py", "adavip/manifeel/reference_diffusion_policy.py",
                 "adavip/manifeel/full_sensor_obs_encoder.py", "scripts/manifeel/eval_input_mask_sensitivity.py"):
        implementation.update((root / name).read_bytes())
    contract = {"checkpoint": str(checkpoint), "checkpoint_bytes": checkpoint.stat().st_size,
                "checkpoint_mtime_ns": checkpoint.stat().st_mtime_ns, "weights": cfg.weights,
                "dataset": manifest_fingerprint(dataset.manifest),
                "normalizer": normalizer_fingerprint(policy.normalizer),
                "horizon": policy.horizon, "n_obs_steps": policy.n_obs_steps,
                "diffusion_steps": policy.noise_scheduler.config.num_train_timesteps,
                "seed": cfg.seed, "repeats": cfg.repeats, "samples": len(dataset),
                "mask_semantics": "post-projection zero L1 tokens plus presence flags; zero late language; L0 native missing tokens",
                "implementation_sha256": implementation.hexdigest(), "runtime": flags}
    cursor_path = output / "cursor.json"
    if cursor_path.exists():
        cursor = json.loads(cursor_path.read_text())
        if cursor["contract"] != contract:
            raise ValueError("Diagnostic continuation contract changed")
        completed = cursor["completed"]
        mode = "r+"
    else:
        completed, mode = 0, "w+"
    arrays = {}
    for name, dtype, shape in (("losses", np.float32, (len(dataset), cfg.repeats, 10)),
                               ("timesteps", np.int64, (len(dataset), cfg.repeats)),
                               ("metadata", np.int64, (len(dataset), 8))):
        arrays[name] = np.lib.format.open_memmap(output / f"{name}.npy", mode=mode, dtype=dtype, shape=shape)
    atomic_json(output / "protocol.json", {**contract, "checkpoint_metadata": metadata,
                "condition_names": list(CONDITION_NAMES),
                "metadata_columns": ["dataset_index", "task_index", "episode_index", "timestep", "source_index",
                                     "full_observation_window", "full_action_window", "valid_action_dimensions"],
                "scope": "all 400 training demonstrations, including boundary anchors; no held-out or simulator score",
                "utility_definition": "masked denoising loss minus full denoising loss; signed, never clipped"})
    print(json.dumps({"event": "diagnostic_ready", "completed": completed, **contract}), flush=True)
    if completed < len(dataset):
        dataset.load_ram_cache()
        loader = DataLoader(Subset(dataset, range(completed, len(dataset))), batch_size=cfg.batch_size,
                            shuffle=False, num_workers=cfg.num_workers, pin_memory=True,
                            persistent_workers=True, prefetch_factor=1, worker_init_fn=initialize_worker,
                            generator=torch.Generator().manual_seed(cfg.seed + 700000))
        started = time.perf_counter()
        for cpu_batch in loader:
            stop_file = os.environ.get("ADAVIP_STOP_FILE")
            if stopped or (stop_file and Path(stop_file).exists()):
                atomic_json(output / "SAFE_STOP.json", {"completed": completed})
                return
            size = len(cpu_batch["action"])
            indices = list(range(completed, completed + size))
            batch = dict_apply(cpu_batch, lambda tensor: tensor.to(policy.device, non_blocking=True))
            losses, steps = paired_mask_losses(policy, batch, indices, cfg.repeats, cfg.seed)
            rows = np.column_stack((indices, *[cpu_batch[name].numpy() for name in
                                               ("task_index", "episode_index", "timestep", "source_index")],
                                    cpu_batch["observation_valid_mask"].all(dim=1).numpy(),
                                    cpu_batch["action_valid_mask"].all(dim=1).numpy(),
                                    cpu_batch["action_loss_mask"].sum(dim=(1, 2)).numpy()))
            end = completed + size
            arrays["losses"][completed:end], arrays["timesteps"][completed:end], arrays["metadata"][completed:end] = losses, steps, rows
            for array in arrays.values():
                array.flush()
            completed = end
            atomic_json(cursor_path, {"contract": contract, "completed": completed})
            if completed == size or completed % (cfg.batch_size * 20) == 0 or completed == len(dataset):
                print(json.dumps({"event": "mask_progress", "completed": completed, "samples": len(dataset),
                                  "seconds": time.perf_counter() - started,
                                  "batch_full_loss": float(losses[:, :, 0].mean()),
                                  "batch_masked_loss": losses[:, :, 1:].mean(axis=(0, 1)).tolist()}), flush=True)
    values = np.asarray(arrays["losses"])
    information = np.asarray(arrays["metadata"])
    task_ids = [spec.task_id for spec in dataset.task_specs]
    by_task = {task: summarize_losses(values[information[:, 1] == index]) for index, task in enumerate(task_ids)}
    averaged = values.mean(axis=1)
    np.save(output / "signed_deltas.npy", averaged[:, 1:] - averaged[:, :1])
    full_windows = (information[:, 5] == 1) & (information[:, 6] == 1)
    report = {"checkpoint_metadata": metadata, "condition_names": list(CONDITION_NAMES),
              "all_anchors": summarize_losses(values), "per_task": by_task,
              "complete_windows": summarize_losses(values[full_windows]),
              "macro_mean_loss": np.mean([row["mean_loss"] for row in by_task.values()], axis=0).tolist(),
              "macro_mean_delta": np.mean([row["mean_delta"] for row in by_task.values()], axis=0).tolist(),
              "caveat": "Offline sensitivity on training demonstrations; mask conditions are OOD for no-dropout L1. Not simulator success."}
    atomic_json(output / "report.json", report)
    lines = ["# Paired input-mask diagnostic", "", f"Checkpoint: `{checkpoint}` ({cfg.weights}).", "",
             "All training anchors; signed delta = masked loss - full loss. This is offline sensitivity, not simulator success.", "",
             "| Task | Full loss | " + " | ".join(CONDITION_NAMES[1:]) + " |",
             "| --- | ---: | " + " | ".join(["---:"] * 9) + " |"]
    for task, row in by_task.items():
        lines.append(f"| {task} | {row['mean_loss'][0]:.7f} | " + " | ".join(f"{value:+.7f}" for value in row["mean_delta"]) + " |")
    (output / "report.md").write_text("\n".join(lines) + "\n")
    atomic_json(output / "COMPLETE.json", {"samples": completed, "repeats": cfg.repeats})
    print(json.dumps({"event": "mask_complete", "samples": completed, "report": str(output / "report.json")}), flush=True)


if __name__ == "__main__":
    main()
