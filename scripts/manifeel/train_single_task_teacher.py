"""Run bounded preflight and an isolated single-task DP teacher experiment."""

from __future__ import annotations

import argparse
import gc
import json
import os
import signal
import time
from pathlib import Path

import dill
import hydra
import torch
from omegaconf import OmegaConf

from adavip.manifeel.single_task_teacher_workspace import SingleTaskTeacherWorkspace
from adavip.manifeel.task_protocol import DEFAULT_TASK_SPECS
from adavip.manifeel.training_runtime import check_dataloader_ipc


def make_config(task_id: str, sensors: str, dropout: bool, output: Path,
                dataset_root: str = "/data/wangzihao/datasets/manifeel"):
    """Build the matched eight-run protocol from one checked-in template."""
    cfg = OmegaConf.load(Path(__file__).resolve().parents[2] / "configs/manifeel/train_single_task_teacher.yaml")
    spec = next(spec for spec in DEFAULT_TASK_SPECS if spec.task_id == task_id)
    if sensors == "five":
        keys = ("front", "side", "wrist", "left_tactile_camera_taxim", "right_tactile_camera_taxim")
    elif sensors == "two":
        keys = ("wrist", "right_tactile_camera_taxim")
    else:
        raise ValueError("sensors must be five or two")
    cfg.shape_meta = {"obs": {**{key: {"shape": [3, 256, 256], "type": "rgb"} for key in keys},
                              "state": {"shape": [7], "type": "low_dim"}}, "action": {"shape": [spec.action_dim]}}
    cfg.task = {"name": task_id, "isaacgym_cfg_name": spec.isaacgym_cfg_name,
                "env_runner": {"_target_": "adavip.manifeel.device_runner.DeviceAwareManifeelRunner",
                               "shape_meta": "${shape_meta}", "isaacgym_cfg_name": spec.isaacgym_cfg_name,
                               "device": "cuda:0", "n_test": 50, "n_test_vis": 2,
                               "test_start_seed": 100000, "max_steps": 500,
                               "n_obs_steps": 2, "n_action_steps": 8, "fps": 10,
                               "past_action": False, "tactile_size": [256, 256]},
                "dataset": {"_target_": "adavip.manifeel.single_task_teacher.SingleTaskTeacherDataset",
                            "shape_meta": "${shape_meta}", "zarr_path": str(Path(dataset_root) / spec.dataset),
                            "horizon": 16, "n_obs_steps": 2, "n_action_steps": 8, "seed": 42,
                            "val_ratio": 0.02, "max_train_episodes": 50}}
    cfg.policy.full_input_probability = 0.6 if dropout else 1.0
    cfg.output_dir = str(output.resolve())
    OmegaConf.resolve(cfg)
    return cfg


def verify_checkpoint(path: Path, batch: dict) -> dict:
    """Strictly restore EMA/normalizer and run a bounded deterministic inference check."""
    payload = torch.load(path, map_location="cpu", pickle_module=dill)
    policy = hydra.utils.instantiate(payload["cfg"].policy)
    policy.load_state_dict(payload["state_dicts"]["ema_model"], strict=True)
    policy.eval()
    # Two denoising steps suffice for an interface/finite-value smoke check.
    policy.num_inference_steps = 2
    obs = {key: value[:1].cpu() for key, value in batch["obs"].items()}
    torch.manual_seed(1050)
    action = policy.predict_action(obs)["action"]
    if action.shape != (1, policy.n_action_steps, policy.action_dim) or not torch.isfinite(action).all():
        raise RuntimeError("Restored EMA action contract failed")
    return {"checkpoint_bytes": path.stat().st_size, "action_shape": list(action.shape),
            "normalizer_fields": list(policy.normalizer.params_dict), "strict_ema_restore": True}


def main() -> None:
    """Preflight each new run before training; continue saved runs without duplication."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=["power_plug_insertion", "usb_insertion", "ball_sorting"])
    parser.add_argument("--sensors", choices=["five", "two"], default="five")
    parser.add_argument("--dropout", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight-steps", type=int, default=40)
    parser.add_argument("--check-inputs", action="store_true")
    parser.add_argument("--config-only", action="store_true")
    args = parser.parse_args()
    cfg = make_config(args.task, args.sensors, args.dropout, args.output)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.config_only:
        OmegaConf.save(cfg, args.output / "config.yaml")
        print(OmegaConf.to_yaml(cfg))
        return
    print(json.dumps(check_dataloader_ipc()), flush=True)
    dataset = hydra.utils.instantiate(cfg.task.dataset)
    sample = torch.utils.data.default_collate([dataset[index] for index in range(8)])
    print(json.dumps({"event": "inputs_checked", "task": args.task, "sensors": args.sensors,
                      "obs_shapes": {key: list(value.shape) for key, value in sample["obs"].items()},
                      "action_shape": list(sample["action"].shape), "train_windows": len(dataset),
                      "validation_windows": len(dataset.get_validation_dataset())}), flush=True)
    if args.check_inputs:
        return
    if not torch.cuda.is_available():
        raise RuntimeError("This training protocol requires an allocated CUDA GPU")
    started = time.perf_counter()
    dataset.load_to_ram(workers=3)
    print(json.dumps({"event": "ram_ready", "seconds": time.perf_counter() - started}), flush=True)
    workspace = None

    def stop(signum: int, frame: object) -> None:
        if workspace is not None:
            workspace.request_stop(signal.Signals(signum).name)

    for signum in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, stop)
    latest = args.output / "checkpoints/latest.ckpt"
    passed = args.output / "preflight/PASSED.json"
    if not latest.exists() and not passed.exists():
        preflight_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        preflight_cfg.output_dir = str(args.output / "preflight")
        workspace = SingleTaskTeacherWorkspace(preflight_cfg)
        workspace.run(dataset, stop_after_steps=args.preflight_steps)
        if workspace.stop_reason:
            (args.output / "SAFE_STOP.json").write_text('{"reason":"preflight_stop"}\n')
            return
        workspace = None
        gc.collect()
        torch.cuda.empty_cache()
        evidence = verify_checkpoint(args.output / "preflight/checkpoints/latest.ckpt", sample)
        records = [json.loads(line) for line in (args.output / "preflight/metrics.jsonl").read_text().splitlines()]
        windows = [record for record in records if record["event"] == "train_step" and record["global_step"] > 1]
        if not windows or any(record["samples_per_second"] <= 0 for record in windows):
            raise RuntimeError("No steady-state preflight throughput evidence")
        affinity_path = Path(os.environ["TEACHER_AFFINITY_LOG"])
        workers = [json.loads(line) for line in affinity_path.read_text().splitlines()]
        trainer_cpus = next(record["trainer_cpus"] for record in records if record["event"] == "startup")
        if len(trainer_cpus) < int(os.environ.get("SLURM_CPUS_PER_TASK", 1)):
            raise RuntimeError("Trainer affinity omits allocated CPUs")
        if not workers or any(record["cpus"] != trainer_cpus for record in workers):
            raise RuntimeError("DataLoader workers did not inherit allocated CPU affinity")
        evidence.update({"steps": args.preflight_steps, "throughput_windows": windows,
                         "trainer_cpus": trainer_cpus, "worker_affinities": workers})
        passed.write_text(json.dumps(evidence, indent=2) + "\n")
        print(json.dumps({"event": "preflight_passed", **evidence}), flush=True)
        gc.collect()
    if latest.exists():
        cfg.resume = str(latest)
    workspace = SingleTaskTeacherWorkspace(cfg)
    workspace.run(dataset)


if __name__ == "__main__":
    main()
