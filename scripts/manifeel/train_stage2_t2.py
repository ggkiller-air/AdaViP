"""Single-H200 T=2 Stage 2 training with resumable checkpoints."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import resource
import signal
import subprocess
import time
from pathlib import Path

import dill
import hydra
import numpy as np
import torch
from torch.utils.data import DataLoader

from adavip.manifeel.full_sensor_workspace import FullSensorWorkspace, OffsetSampler, normalizer_fingerprint
from adavip.manifeel.stage2_data import load_complete_ram_cache, make_stage2_dataset
from adavip.manifeel.stage2_policy import Stage2Policy


SNAPSHOT_EPOCHS = frozenset((1, 6, 10, 20, 50, 100, 200, 300, 400, 500, 600))
STOP_REQUESTED = False


def request_stop(signum: int, frame: object) -> None:
    """Finish the active optimizer update before checkpointing on Slurm signal."""
    global STOP_REQUESTED
    STOP_REQUESTED = True


def stage_for_epoch(epoch: int) -> str:
    """Map one-based epoch to the three explicit trainability phases."""
    return "A" if epoch == 1 else "B" if epoch <= 6 else "C"


def set_stage_trainability(policy: Stage2Policy, stage: str) -> None:
    """Freeze the inherited path for A/B and all action modules for A."""
    if stage not in ("A", "B", "C"):
        raise ValueError(stage)
    policy.configure_encoders(stage)
    for module in (policy.z_projection, policy.utility_projection, policy.film_hypernets,
                   policy.query_projection, policy.cross_attention, policy.fusion_norm,
                   policy.fusion_ff, policy.model):
        module.requires_grad_(stage != "A")
    policy.utility_predictor.requires_grad_(True)
    policy.normalizer.requires_grad_(False)


def make_optimizer_scheduler(policy: Stage2Policy, steps_per_epoch: int,
                             epochs: int = 600) -> tuple[torch.optim.Optimizer, object]:
    """Use AdamW with cosine decay and a one-epoch encoder unfreeze ramp."""
    encoder_groups = policy.encoder_optimizer_groups()
    encoder = list(policy.obs_encoder.parameters())
    if policy.action_encoder is not None:
        encoder += list(policy.action_encoder.parameters())
    utility = list(policy.utility_predictor.parameters())
    taken = {id(item) for item in encoder + utility}
    action = [item for name, item in policy.named_parameters()
              if id(item) not in taken and not name.startswith("normalizer.") and name != "_dummy_variable"]
    optimizer = torch.optim.AdamW([
        *encoder_groups,
        {"params": utility, "lr": 1e-4, "name": "utility"},
        {"params": action, "lr": 1e-4, "name": "action"},
    ], betas=(0.95, 0.999), eps=1e-8, weight_decay=1e-6)
    total = steps_per_epoch * epochs
    c_start = steps_per_epoch * (1 if policy.is_dual_encoder else 6)

    def shared(step: int) -> float:
        if step < 500:
            return max(1e-3, (step + 1) / 500)
        progress = min(1.0, (step - 500) / max(1, total - 500))
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

    def encoder_schedule(step: int) -> float:
        if step < c_start:
            return 0.0
        return shared(step) * min(1.0, (step - c_start + 1) / 500)

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=([encoder_schedule] * len(encoder_groups) + [shared, shared]),
    )
    return optimizer, scheduler


def gpu_status() -> dict[str, int]:
    """Read one bounded NVML CLI sample without touching other GPU processes."""
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True, timeout=8,
    )
    rows = result.stdout.strip().splitlines()
    if len(rows) != 1:
        raise RuntimeError("Stage 2 requires exactly one visible H200")
    utilization, memory = (int(value.strip()) for value in rows[0].split(","))
    return {"gpu_utilization_percent": utilization, "gpu_memory_mib": memory}


def cgroup_memory_bytes() -> int | None:
    """Report the job's charged RAM when the node exposes its cgroup counter."""
    for name in ("/sys/fs/cgroup/memory.current", "/sys/fs/cgroup/memory/memory.usage_in_bytes"):
        path = Path(name)
        if path.is_file():
            return int(path.read_text().strip())
    return None


def atomic_checkpoint(path: Path, payload: dict) -> None:
    """Replace only a complete checkpoint in the isolated Stage 2 directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial")
    torch.save(payload, temporary, pickle_module=dill)
    os.replace(temporary, path)


def rng_state() -> dict:
    """Capture main-process random streams for exact optimizer continuation."""
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()}


def restore_rng(state: dict) -> None:
    """Restore the random streams saved after the last complete optimizer step."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state_all(state["cuda"])


def update_ema(ema: Stage2Policy, policy: Stage2Policy, decay: float) -> None:
    """Update every trainable parameter and copy immutable buffers."""
    with torch.no_grad():
        for target, source in zip(ema.parameters(), policy.parameters()):
            target.lerp_(source, 1 - decay)
        for target, source in zip(ema.buffers(), policy.buffers()):
            target.copy_(source)


def restore_policy_state(policy: Stage2Policy, state: dict, device: torch.device) -> None:
    """Restore a policy and keep the custom Normalizer on the model device."""
    policy.load_state_dict(state, strict=True)
    # Diffusion Policy's Normalizer rebuilds its ParameterDict from the state
    # tensors, so loading a CPU checkpoint after policy.to(cuda) moves it to CPU.
    policy.normalizer.to(device)
    if any(parameter.device != device for parameter in policy.normalizer.parameters()):
        raise RuntimeError("Restored Normalizer remains on the wrong device")


def load_teacher(path: str, label_directory: str) -> object:
    """Read pinned E500 EMA and compare all 48 saved Normalizer tensors."""
    workspace = FullSensorWorkspace.create_from_checkpoint(path, map_location="cpu")
    if workspace.epoch != 500 or workspace.global_step != 547400 or workspace.batch_in_epoch != 0:
        raise ValueError("Teacher checkpoint is not the selected complete E500")
    if workspace.ema_model is None:
        raise ValueError("Teacher checkpoint lacks EMA policy")
    normalizer_path = Path(label_directory) / "metadata" / "teacher_normalizer.npz"
    state = workspace.ema_model.normalizer.state_dict()
    with np.load(normalizer_path, allow_pickle=False) as archived:
        if set(archived.files) != set(state):
            raise ValueError("Teacher and label Normalizer tensor keys differ")
        for key, value in state.items():
            if not np.array_equal(value.detach().cpu().numpy(), archived[key]):
                raise ValueError(f"Teacher and label Normalizer differ: {key}")
    return workspace


def build_policy(teacher: object, utility_rms: np.ndarray,
                 encoder_strategy: str = "s0") -> Stage2Policy:
    """Create a fresh Stage 2 DP and strictly inherit only encoder/normalizer."""
    cfg = teacher.cfg.policy
    encoder = hydra.utils.instantiate(cfg.obs_encoder)
    scheduler = hydra.utils.instantiate(cfg.noise_scheduler)
    policy = Stage2Policy(
        obs_encoder=encoder, noise_scheduler=scheduler,
        utility_rms=torch.from_numpy(utility_rms.astype(np.float32)),
        encoder_strategy=encoder_strategy,
        horizon=cfg.horizon, n_obs_steps=cfg.n_obs_steps,
        n_action_steps=cfg.n_action_steps, num_inference_steps=cfg.num_inference_steps,
        diffusion_step_embed_dim=cfg.diffusion_step_embed_dim,
        down_dims=tuple(cfg.down_dims), kernel_size=cfg.kernel_size,
        n_groups=cfg.n_groups, cond_predict_scale=cfg.cond_predict_scale,
    )
    policy.initialize_from_teacher(teacher.ema_model)
    return policy


def parse_args() -> argparse.Namespace:
    """Parse the frozen single-GPU run contract."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--cache-config", required=True)
    parser.add_argument("--dataset-root", default="/data/wangzihao/datasets/manifeel")
    parser.add_argument("--task-embeddings", default="/data/wangzihao/outputs/manifeel/task_embeddings/clip_vit_b32_9task.npz")
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--preflight-steps", type=int, default=0)
    parser.add_argument("--continue-after-preflight", action="store_true")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=600)
    return parser.parse_args()


def main() -> None:
    """Train all complete labeled windows with stage gates and atomic resume."""
    args = parse_args()
    if args.batch_size != 32 or args.epochs != 600 or args.workers < 0:
        raise ValueError("Official Stage 2 protocol is batch32/600 epochs")
    if args.continue_after_preflight and (args.preflight_steps < 7 or args.resume):
        raise ValueError("Combined run needs at least seven preflight steps and no resume")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or "H200" not in torch.cuda.get_device_name(0):
        raise RuntimeError("Stage 2 requires exactly one NVIDIA H200")
    signal.signal(signal.SIGUSR1, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    dataset = make_stage2_dataset(args.labels, args.dataset_root, args.task_embeddings)
    label_teacher = dataset.store.manifest["teacher"]
    if label_teacher["metadata"] != {"epoch": 500, "global_step": 547400,
                                     "batch_in_epoch": 0, "weights": "ema"}:
        raise ValueError("Label store belongs to a different Teacher")
    utility = np.asarray(dataset.store.utility, dtype=np.float64)
    utility_rms = np.maximum(np.sqrt(np.mean(np.square(utility), axis=0)), 1e-6)
    cache_bytes = load_complete_ram_cache(dataset.source, args.cache, args.cache_config)
    teacher = load_teacher(args.teacher, args.labels)
    policy = build_policy(teacher, utility_rms)
    del teacher
    device = torch.device("cuda:0")
    policy.to(device)
    ema = copy.deepcopy(policy).eval().requires_grad_(False)
    steps_per_epoch = math.ceil(len(dataset) / args.batch_size)
    optimizer, scheduler = make_optimizer_scheduler(policy, steps_per_epoch, args.epochs)
    initial_model = ({key: value.detach().cpu().clone() for key, value in policy.state_dict().items()}
                     if args.continue_after_preflight else None)
    initial_rng = rng_state() if args.continue_after_preflight else None
    contract = {"teacher": str(Path(args.teacher).resolve()), "labels_sha256": dataset.store.manifest["sha256"],
                "cache": str(Path(args.cache).resolve()),
                "cache_config": str(Path(args.cache_config).resolve()), "batch_size": args.batch_size,
                "epochs": args.epochs, "rms": utility_rms.tolist(), "utility_weight": 0.05,
                "normalizer": normalizer_fingerprint(policy.normalizer)}
    contract_hash = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    epoch, batch_cursor, global_step = 1, 0, 0
    if args.resume:
        saved = torch.load(args.resume, map_location="cpu", pickle_module=dill)
        if saved["contract_hash"] != contract_hash:
            raise ValueError("Resume contract differs from the original Stage 2 run")
        restore_policy_state(policy, saved["model"], device)
        restore_policy_state(ema, saved["ema"], device)
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        epoch, batch_cursor, global_step = saved["epoch"], saved["batch_cursor"], saved["global_step"]
        restore_rng(saved["rng"])
    (output / "contract.json").write_text(json.dumps(contract, indent=2) + "\n")

    def save(tag: str) -> None:
        payload = {"model": policy.state_dict(), "ema": ema.state_dict(),
                   "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                   "epoch": epoch, "batch_cursor": batch_cursor, "global_step": global_step,
                   "rng": rng_state(), "contract_hash": contract_hash, "contract": contract,
                   "stage": stage_for_epoch(min(epoch, args.epochs))}
        atomic_checkpoint(output / "checkpoints" / f"{tag}.ckpt", payload)

    print(json.dumps({"event": "stage2_ready", "gpu": torch.cuda.get_device_name(0),
                      "gpu_status": gpu_status(), "windows": len(dataset), "cache_bytes": cache_bytes,
                      "steps_per_epoch": steps_per_epoch, "utility_rms": utility_rms.tolist(),
                      "next_epoch": epoch, "next_batch": batch_cursor,
                      "available_cpus": len(os.sched_getaffinity(0)), "workers": args.workers,
                      "ram_peak_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      "cgroup_ram_bytes": cgroup_memory_bytes()}), flush=True)
    preflight_left = args.preflight_steps
    while epoch <= args.epochs:
        stage = stage_for_epoch(epoch)
        set_stage_trainability(policy, stage)
        policy.train()
        generator = torch.Generator().manual_seed(42 + epoch)
        sampler = dataset.make_task_balanced_sampler(generator)
        if batch_cursor:
            sampler = OffsetSampler(sampler, batch_cursor * args.batch_size)
        loader_options = {"batch_size": args.batch_size, "sampler": sampler,
                          "num_workers": args.workers, "pin_memory": True,
                          "persistent_workers": args.workers > 0}
        if args.workers:
            loader_options["prefetch_factor"] = 2
            loader_options["multiprocessing_context"] = "fork"
        loader = DataLoader(dataset, **loader_options)
        start = time.monotonic()
        previous_end = start
        data_wait_seconds = 0.0
        sums = torch.zeros(9, device=device, dtype=torch.float64)
        squares = torch.zeros(9, device=device, dtype=torch.float64)
        metric_count = 0
        restart_after_preflight = False
        for batch in loader:
            data_wait_seconds += time.monotonic() - previous_end
            if preflight_left:
                preflight_index = args.preflight_steps - preflight_left
                stage = "A" if preflight_index < 2 else "B" if preflight_index < 6 else "C"
                set_stage_trainability(policy, stage)
            batch = {key: ({sensor: value.to(device, non_blocking=True) for sensor, value in item.items()}
                           if key == "obs" else item.to(device, non_blocking=True))
                     for key, item in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            result = policy.training_losses(batch, stage)
            loss = result["utility_loss"] if stage == "A" else result["action_loss"] + 0.05 * result["utility_loss"]
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite Stage 2 loss")
            loss.backward()
            if preflight_left and stage == "C":
                if not any(parameter.grad is not None and torch.count_nonzero(parameter.grad).item() > 0
                           for parameter in policy.obs_encoder.parameters()):
                    raise RuntimeError("Action loss did not reach the Stage 2 encoder")
            grad_norm = torch.nn.utils.clip_grad_norm_([p for p in policy.parameters() if p.requires_grad], 1.0)
            if not torch.isfinite(grad_norm):
                raise FloatingPointError("Nonfinite Stage 2 gradient")
            optimizer.step()
            scheduler.step()
            update_ema(ema, policy, 0.9999)
            global_step += 1
            batch_cursor += 1
            error = (result["u_hat"].detach() - result["utility_target"]).float()
            sums += error.abs().sum(dim=0).double()
            squares += error.square().sum(dim=0).double()
            metric_count += len(error)
            if global_step % 20 == 0 or preflight_left == 1:
                record = {"event": "train", "epoch": epoch, "stage": stage,
                          "batch_cursor": batch_cursor, "global_step": global_step,
                          "action_loss": float(result["action_loss"].detach()),
                          "utility_loss": float(result["utility_loss"].detach()),
                          "gradient_norm": float(grad_norm),
                          "lr": {group["name"]: group["lr"] for group in optimizer.param_groups},
                          "utility_mae_9d": (sums / metric_count).cpu().tolist(),
                          "utility_rmse_9d": (squares / metric_count).sqrt().cpu().tolist(),
                          "samples_per_second": metric_count / max(1e-6, time.monotonic() - start),
                          "data_wait_seconds": data_wait_seconds,
                          "ram_peak_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                          "cgroup_ram_bytes": cgroup_memory_bytes(),
                          "gpu_allocated_bytes": torch.cuda.memory_allocated(),
                          "gpu_peak_bytes": torch.cuda.max_memory_allocated(),
                          "gpu_status": gpu_status() if global_step % 100 == 0 or preflight_left == 1 else None}
                metrics_file = "preflight_metrics.jsonl" if preflight_left else "metrics.jsonl"
                with (output / metrics_file).open("a") as stream:
                    stream.write(json.dumps(record) + "\n")
                print(json.dumps(record), flush=True)
            if preflight_left:
                preflight_left -= 1
                if preflight_left == 0:
                    report = {"event": "preflight_complete", "epoch": epoch,
                              "batch_cursor": batch_cursor, "global_step": global_step,
                              "gpu_status": gpu_status(), "ram_cache_bytes": cache_bytes,
                              "last_action_loss": float(result["action_loss"].detach()),
                              "last_utility_loss": float(result["utility_loss"].detach()),
                              "gradient_norm": float(grad_norm),
                              "data_wait_seconds": data_wait_seconds,
                              "ram_peak_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                              "cgroup_ram_bytes": cgroup_memory_bytes(),
                              "gpu_peak_bytes": torch.cuda.max_memory_allocated(),
                              "samples_per_second": metric_count / max(1e-6, time.monotonic() - start)}
                    (output / "preflight_report.json").write_text(json.dumps(report, indent=2) + "\n")
                    print(json.dumps(report), flush=True)
                    if not args.continue_after_preflight:
                        return
                    if (report["samples_per_second"] < 20 or
                            report["data_wait_seconds"] / max(1e-6, time.monotonic() - start) > 0.75 or
                            report["gpu_peak_bytes"] > 0.9 * torch.cuda.get_device_properties(0).total_memory):
                        raise RuntimeError("Stage 2 preflight throughput or memory gate failed")
                    restore_policy_state(policy, initial_model, device)
                    restore_policy_state(ema, initial_model, device)
                    optimizer, scheduler = make_optimizer_scheduler(policy, steps_per_epoch, args.epochs)
                    restore_rng(initial_rng)
                    del initial_model
                    initial_model = None
                    epoch, batch_cursor, global_step = 1, 0, 0
                    restart_after_preflight = True
                    print(json.dumps({"event": "formal_start_after_preflight", "report": str(output / "preflight_report.json")}), flush=True)
                    break
            if STOP_REQUESTED:
                save("latest")
                (output / "SAFE_STOP.json").write_text(json.dumps({"epoch": epoch,
                    "batch_cursor": batch_cursor, "global_step": global_step}) + "\n")
                print(json.dumps({"event": "safe_stop", "epoch": epoch,
                                  "batch_cursor": batch_cursor, "global_step": global_step}), flush=True)
                return
            previous_end = time.monotonic()
        del loader
        if restart_after_preflight:
            continue
        completed = epoch
        epoch += 1
        batch_cursor = 0
        save("latest")
        if completed in SNAPSHOT_EPOCHS:
            save(str(completed))
        print(json.dumps({"event": "epoch_complete", "epoch": completed,
                          "global_step": global_step, "utility_mae_9d": (sums / metric_count).cpu().tolist(),
                          "utility_rmse_9d": (squares / metric_count).sqrt().cpu().tolist()}), flush=True)
    (output / "COMPLETE.json").write_text(json.dumps({"epoch": args.epochs,
        "global_step": global_step, "contract_hash": contract_hash}) + "\n")


if __name__ == "__main__":
    main()
