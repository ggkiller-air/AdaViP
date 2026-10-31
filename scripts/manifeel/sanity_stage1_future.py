"""Bounded, real-data H200 checks and calibration for the four new Stage-1 groups."""

from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import time
from pathlib import Path

import hydra
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from adavip.manifeel.full_sensor_workspace import normalizer_fingerprint
from adavip.manifeel.stage1_future import gradient_diagnostics, stage1_losses
from adavip.manifeel.stage1_future_calibration import calibrate_future_weight
from adavip.manifeel.stage1_future_dataset import initialize_future_worker
from adavip.manifeel.stage1_future_workspace import Stage1FutureWorkspace, future_source_fingerprint
from diffusion_policy.common.pytorch_util import dict_apply
from scripts.manifeel.language_runtime import configure_profile


def main() -> None:
    """Check real batch32 updates, all five ResNets, target EMA and checkpoint IO."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=4)
    args = parser.parse_args()
    if not torch.cuda.is_available() or "H200" not in torch.cuda.get_device_name(0):
        raise RuntimeError("Sanity requires an allocated H200")
    root = Path(__file__).resolve().parents[2]
    cfg_base = OmegaConf.load(root / "configs/manifeel/train_stage1_future.yaml")
    args.output.mkdir(parents=True, exist_ok=True)
    flags = configure_profile("tf32_deterministic")
    device = torch.device("cuda:0")
    report = {"gpu": torch.cuda.get_device_name(0), "runtime": flags,
              "source_fingerprint": future_source_fingerprint(),
              "trainer_affinity": sorted(os.sched_getaffinity(0)), "experiments": {}}
    shared = hydra.utils.instantiate(cfg_base.dataset)
    started = time.perf_counter()
    shared.load_ram_cache()
    report["cache_load_seconds"] = time.perf_counter() - started
    # Independent worker/IPC performance check using the same data and initialization.
    affinity_dir = args.output / "affinity"
    os.environ["STAGE1_FUTURE_AFFINITY_DIR"] = str(affinity_dir)
    loader = DataLoader(shared, batch_size=32, num_workers=8, pin_memory=True,
                        worker_init_fn=initialize_future_worker, prefetch_factor=1,
                        generator=torch.Generator().manual_seed(940042))
    started = time.perf_counter()
    iterator = iter(loader)
    for _ in range(3):
        batch = next(iterator)
        assert len(batch["action"]) == 32
    report["worker_test_seconds"] = time.perf_counter() - started
    del iterator, loader, batch
    worker_records = [json.loads(p.read_text()) for p in affinity_dir.glob("worker_*.json")]
    assert len(worker_records) == 8
    assert len({tuple(r["actual_affinity"]) for r in worker_records}) == 8
    assert all(set(r["actual_affinity"]).issubset(report["trainer_affinity"]) for r in worker_records)
    report["workers"] = worker_records
    initial_hashes = set()
    for experiment in "BCDE":
        print(json.dumps({"event": "sanity_start", "experiment": experiment}), flush=True)
        cfg = copy.deepcopy(cfg_base)
        cfg.experiment, cfg.output_dir = experiment, str(args.output / experiment)
        OmegaConf.resolve(cfg)
        output = Path(cfg.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(cfg, output / "config.yaml", resolve=True)
        dataset = hydra.utils.instantiate(cfg.dataset)
        dataset.groups, dataset.verified_ram_cache = shared.groups, True
        workspace = Stage1FutureWorkspace(cfg, cfg.output_dir)
        initial_hashes.add(workspace.policy_initialization)
        policy, auxiliary = workspace.model.to(device), workspace.future.to(device)
        # The policy EMA is not needed for the GPU compute checks; keep it on CPU.
        policy.set_normalizer(dataset.get_normalizer())
        policy.normalizer.to(device)
        workspace.ema_model.set_normalizer(dataset.get_normalizer())
        noise = torch.Generator(device=device).manual_seed(300042)
        workspace._noise_generator = noise
        before = normalizer_fingerprint(policy)
        calibration = calibrate_future_weight(policy, auxiliary, dataset, device, batch_size=32,
                                              batches=cfg.future.calibration_batches)
        assert normalizer_fingerprint(policy) == before
        workspace.future_weight, workspace.calibration = calibration["weight"], calibration
        (output / "calibration.json").write_text(json.dumps(calibration, indent=2) + "\n")
        calibration["policy_unchanged"] = True
        entries = []
        dataset_indices = []
        previous = 0
        for length in dataset.task_lengths:
            dataset_indices.extend(range(previous + 4, previous + 8))
            previous += length
        dataset_indices = dataset_indices[:32]
        batch = torch.utils.data.default_collate([dataset[i] for i in dataset_indices])
        assert (batch["future_index"][batch["future_valid"]] ==
                batch["future_anchor"][batch["future_valid"]] + 4).all()
        assert (batch["future_index"][batch["future_valid"]] <
                batch["future_episode_end"][batch["future_valid"]]).all()
        batch = dict_apply(batch, lambda tensor: tensor.to(device))
        parameter_sets = [set(p.data_ptr() for p in head.parameters()) for head in auxiliary.heads.values()]
        assert all(left.isdisjoint(right) for i, left in enumerate(parameter_sets)
                   for right in parameter_sets[i + 1:])
        torch.cuda.reset_peak_memory_stats()
        for step in range(args.steps):
            started = time.perf_counter()
            workspace.optimizer.zero_grad(set_to_none=True)
            sample = policy.prepare_denoising_sample(batch, generator=noise)
            terms = stage1_losses(policy, auxiliary, batch, sample)
            assert terms["z_t"].shape == (32, 256)
            if step == 0:
                with torch.no_grad():
                    original = policy.compute_loss_per_sample(
                        batch, torch.zeros(32, dtype=torch.long, device=device), sample)
                torch.testing.assert_close(terms["action_per_sample"], original, rtol=0, atol=0)
                del original
            diagnostics = gradient_diagnostics(terms, policy, workspace.future_weight,
                                               include_resnets=True) if experiment != "A" else {}
            if experiment != "A":
                assert all(v > 0 for v in diagnostics["future_resnet_first_weight_grad_norms"].values())
            old_online = {key: next(m.parameters()).detach().clone()
                          for key, m in policy.obs_encoder.image_encoders.items()}
            old_target = {key: next(m.parameters()).detach().clone()
                          for key, m in auxiliary.target_encoders.items()}
            total = terms["action"] + workspace.future_weight * terms["future"]
            total.backward()
            assert all(p.grad is None for p in auxiliary.target_encoders.parameters())
            grad_norm = torch.nn.utils.clip_grad_norm_(workspace.trainable_parameters(), 1,
                                                       error_if_nonfinite=True)
            workspace.optimizer.step()
            auxiliary.update_targets(policy.obs_encoder)
            assert all(not torch.equal(value, next(policy.obs_encoder.image_encoders[key].parameters()))
                       for key, value in old_online.items())
            for key, old in old_target.items():
                expected = old * 0.996 + next(policy.obs_encoder.image_encoders[key].parameters()) * 0.004
                torch.testing.assert_close(next(auxiliary.target_encoders[key].parameters()), expected)
            torch.cuda.synchronize()
            record = {"step": step + 1, "action_loss": float(terms["action"].detach()),
                      "future_loss": float(terms["future"].detach()), "total_loss": float(total.detach()),
                      "per_sensor_future_loss": {k: float(v.detach()) for k, v in terms["sensors"].items()},
                      "grad_norm": float(grad_norm), "seconds": time.perf_counter() - started,
                      "target_ema_updates": int(auxiliary.ema_updates), **diagnostics}
            entries.append(record)
            with (output / "steps.jsonl").open("a") as stream:
                stream.write(json.dumps(record, allow_nan=False) + "\n")
            del terms, total, sample, old_online, old_target
        workspace.global_step = args.steps
        workspace._save_latest()
        checkpoint = output / "checkpoints/latest.ckpt"
        # Full checkpoint serialization was exercised; exact resume is also a CPU regression test.
        item = {"passed": True, "parameters": workspace.parameter_counts(),
                "initialization": workspace.policy_initialization, "future_weight": workspace.future_weight,
                "calibration": calibration, "steps": entries,
                "checkpoint_bytes": checkpoint.stat().st_size,
                "peak_gpu_gib": torch.cuda.max_memory_allocated() / 1024**3}
        report["experiments"][experiment] = item
        (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps({"event": "sanity_passed", "experiment": experiment,
                          "future_weight": workspace.future_weight, "parameters": item["parameters"]}), flush=True)
        del workspace, policy, auxiliary, dataset, batch
        gc.collect()
        torch.cuda.empty_cache()
    assert len(initial_hashes) == 1
    report["identical_initialization"] = True
    report["passed"] = True
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (args.output / "PASS.json").write_text(json.dumps({"passed": True, "groups": "BCDE"}) + "\n")


if __name__ == "__main__":
    main()
