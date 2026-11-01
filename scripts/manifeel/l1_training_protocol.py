"""Reject shared training settings that differ from the original L1 protocol."""

from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf


def validate_l1_training_protocol(cfg: object, baseline_path: Path | None = None) -> dict:
    """Compare resolved settings, allowing only the declared experimental branches."""
    if baseline_path is None:
        baseline_path = Path(__file__).resolve().parents[2] / "configs/manifeel/train_language_conditioning.yaml"
    baseline = OmegaConf.to_container(OmegaConf.load(baseline_path), resolve=True)
    actual = OmegaConf.to_container(cfg, resolve=True)
    fields = ("variant", "dataset_root", "task_embedding_path", "horizon", "n_obs_steps",
              "n_action_steps", "shape_meta", "optimizer", "ema", "dataloader",
              "val_dataloader", "training", "diagnostics", "runtime")
    expected = {key: baseline[key] for key in fields}
    expected["variant"] = "l1"
    errors = [key for key in fields if actual[key] != expected[key]]
    is_future = "experiment" in actual
    dataset = dict(actual["dataset"])
    policy = dict(actual["policy"])
    encoder = dict(policy["obs_encoder"])
    if is_future:
        if actual["experiment"] not in ("B", "C", "D", "E") or actual["future"]["offset"] != 4:
            errors.append("future experiment")
        if dataset.pop("_target_") != "adavip.manifeel.stage1_future_dataset.Stage1FutureDataset":
            errors.append("dataset._target_")
        for key in ("experiment", "future_offset", "raw_resolution"):
            dataset.pop(key)
        dataset["_target_"] = baseline["dataset"]["_target_"]
        probability = 1.0
    else:
        if encoder["_target_"] != "adavip.manifeel.l1_dropout.L1DropoutObsEncoder":
            errors.append("policy.obs_encoder._target_")
        encoder["_target_"] = baseline["policy"]["obs_encoder"]["_target_"]
        probability = 0.9
    if policy["full_input_probability"] != probability:
        errors.append("policy.full_input_probability")
    policy["full_input_probability"] = baseline["policy"]["full_input_probability"]
    policy["obs_encoder"] = encoder
    if dataset != baseline["dataset"]:
        errors.append("dataset")
    if policy != baseline["policy"]:
        errors.append("policy")
    if errors:
        raise ValueError("Original L1 protocol mismatch: " + ", ".join(errors))
    return {"matched": True, "baseline": str(baseline_path), "shared_fields": list(fields),
            "batch_size": actual["dataloader"]["batch_size"],
            "num_epochs": actual["training"]["num_epochs"],
            "base_lr": actual["optimizer"]["lr"],
            "checkpoint_epochs": actual["training"]["checkpoint_epochs"]}
