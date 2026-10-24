"""Bounded real-H200 checks and batch selection before formal Teacher training."""

from __future__ import annotations

import copy
import gc
import json
import time
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset

from adavip.manifeel.full_sensor_workspace import manifest_fingerprint, normalizer_fingerprint
from adavip.manifeel.stage1_v2_schema import IMAGE_KEYS
from adavip.manifeel.stage1_v2_training import (
    Stage1V2Workspace, capture_rng, choose_future_weights, gradient_diagnostics,
    restore_rng, source_fingerprint,
)
from diffusion_policy.common.pytorch_util import dict_apply


def run_preflight(cfg, report_path: Path) -> tuple:
    """Check actual tensors, leakage, gradients, diffusion, resume and H200 throughput."""
    if not torch.cuda.is_available() or "H200" not in torch.cuda.get_device_name(0):
        raise RuntimeError("The batch comparison must execute on the allocated H200")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    report_path.parent.mkdir(parents=True, exist_ok=True)
    dataset = hydra.utils.instantiate(cfg.dataset)
    if len(dataset.task_specs) != 9 or sum(map(len, dataset.train_episodes)) != 400:
        raise ValueError("Preflight requires exactly nine tasks and 400 demonstrations")
    normalizer = dataset.get_normalizer(mode=str(cfg.training.normalizer_mode))
    normalizer_hash = normalizer_fingerprint(normalizer)
    task_starts = np.cumsum([0] + dataset.task_lengths[:-1]).tolist()
    indices = [int(start + offset % length) for offset in range(4)
               for start, length in zip(task_starts, dataset.task_lengths)]
    cpu_batch = next(iter(DataLoader(Subset(dataset, indices), batch_size=32)))
    device = torch.device("cuda:0")
    batch = dict_apply(cpu_batch, lambda value: value.to(device))
    tiny_batch = dict_apply(batch, lambda value: value[:1])
    workspace = Stage1V2Workspace(cfg, output_dir=str(report_path.parent / "resume_check"))
    workspace.model.set_normalizer(normalizer)
    model = workspace.model.to(device)
    print(json.dumps({"event": "preflight_model_ready", "parameters": model.parameter_counts(),
                      "windows": len(dataset), "task_windows": dataset.task_lengths}), flush=True)
    for key in IMAGE_KEYS:
        for name, value in model.image_encoders[key].state_dict().items():
            if not torch.equal(value, model.target_image_encoders[key].state_dict()[name]):
                raise AssertionError("EMA target was not initialized from this run's online encoder")

    model.eval()
    with torch.no_grad():
        observations = model.normalizer.normalize(tiny_batch["obs"])
        history = model.normalizer["action"].normalize(tiny_batch["action_history"])
        history *= tiny_batch["action_dimension_mask"].unsqueeze(1)
        for condition in range(1, 9):
            presence = model.condition_presence[condition:condition + 1]
            original = model._state_token(observations, history,
                                          tiny_batch["action_history_valid_mask"], presence)
            modified = {key: value.clone() for key, value in observations.items()}
            for key, present in zip((*IMAGE_KEYS, "state", "task_embedding"), presence[0]):
                if not present:
                    modified[key].normal_(mean=3, std=2)
            changed = model._state_token(modified, history,
                                         tiny_batch["action_history_valid_mask"], presence)
            for original_value, changed_value in zip(original, changed):
                torch.testing.assert_close(original_value, changed_value, rtol=0, atol=0)
        sample = model.prepare_denoising_sample(tiny_batch)
        conditions = torch.zeros(1, dtype=torch.long, device=device)
        baseline = model.compute_loss_with_metrics(tiny_batch, conditions, sample)
        changed_batch = dict_apply(tiny_batch, lambda value: value.clone())
        changed_batch["action"][:, 1:5].normal_(mean=10)
        for value in changed_batch["future_obs"].values():
            value.normal_()
        changed = model.compute_loss_with_metrics(changed_batch, conditions, sample)
        torch.testing.assert_close(baseline["z"], changed["z"], rtol=0, atol=0)
        torch.testing.assert_close(baseline["action_loss"], changed["action_loss"], rtol=0, atol=0)
        prediction = model.predict_action({
            **tiny_batch["obs"], "action_history": tiny_batch["action_history"],
            "action_history_valid_mask": tiny_batch["action_history_valid_mask"],
            "action_dimension_mask": tiny_batch["action_dimension_mask"],
        })
        if prediction.shape != (1, 8, 7) or not torch.isfinite(prediction).all():
            raise AssertionError("Full 100-step diffusion inference failed")
    print('GPU mask/leakage/100-step diffusion checks passed', flush=True)
    del original, changed, baseline, sample, observations, history, changed_batch
    model.train()
    audits = []
    calibration_batch = dict_apply(batch, lambda value: value[:8])
    for _ in range(3):
        losses = model.compute_loss_with_metrics(calibration_batch)
        audits.append(gradient_diagnostics(model, losses))
        del losses
    lambda_vision, lambda_tactile = choose_future_weights(
        audits, float(cfg.training.future_gradient_fraction),
    )
    print(json.dumps({"event": "calibration", "audits": audits,
                      "lambda_vision": lambda_vision, "lambda_tactile": lambda_tactile}), flush=True)

    def update(current_batch):
        workspace.optimizer.zero_grad(set_to_none=True)
        losses = model.compute_loss_with_metrics(current_batch)
        total = losses["action_loss"] + lambda_vision * losses["future_vision_loss"] \
            + lambda_tactile * losses["future_tactile_loss"]
        if not torch.isfinite(total):
            raise FloatingPointError("Nonfinite H200 preflight loss")
        total.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        workspace.optimizer.step()
        model.update_target_encoder()
        return {name: float(losses[name].detach()) for name in (
            "action_loss", "future_vision_loss", "future_tactile_loss",
            "vision_latent_variance", "tactile_latent_variance",
        )}, float(norm)

    update(calibration_batch)
    required_modules = ("image_encoders", "image_temporal", "proprio_frame", "proprio_temporal",
                        "language_projection", "action_history", "state_tokenizer", "cross_attention",
                        "model", "future_vision_head", "future_tactile_head")
    for name in required_modules:
        if not any(parameter.grad is not None and parameter.grad.abs().sum() > 0
                   for parameter in getattr(model, name).parameters()):
            raise AssertionError(f"Missing training gradient: {name}")
    if any(parameter.grad is not None for parameter in model.target_image_encoders.parameters()):
        raise AssertionError("EMA targets received gradients")
    workspace.scheduler = torch.optim.lr_scheduler.StepLR(workspace.optimizer, step_size=100)
    workspace.dataset = dataset
    workspace.lambda_vision, workspace.lambda_tactile = lambda_vision, lambda_tactile
    workspace.provenance = {"preflight": True}
    workspace.global_step = 1
    workspace.sample_cursor = 8
    checkpoint = workspace.save("exact_resume_probe")
    rng = capture_rng()
    expected_metrics, expected_norm = update(calibration_batch)
    reference = {
        name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    saved_rng = workspace.restore(str(checkpoint), device)
    restore_rng(saved_rng)
    actual_metrics, actual_norm = update(calibration_batch)
    for name, parameter in model.named_parameters():
        if name in reference:
            torch.testing.assert_close(parameter.detach().cpu(), reference[name], rtol=0, atol=0)
    if actual_metrics != expected_metrics or actual_norm != expected_norm:
        raise AssertionError("Exact GPU optimizer/RNG resume did not reproduce the next update")
    if normalizer_fingerprint(model.normalizer) != normalizer_hash:
        raise AssertionError("Normalizer changed during training/resume")
    del reference, saved_rng, rng
    print('GPU gradients/normalizer/exact optimizer resume checks passed', flush=True)
    workspace.optimizer.zero_grad(set_to_none=True)
    benchmarks = []
    for size in (8, 16, 32):
        current_batch = dict_apply(batch, lambda value: value[:size])
        record = {"batch_size": size}
        try:
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            for _ in range(2):
                update(current_batch)
            torch.cuda.synchronize()
            started = time.perf_counter()
            metrics = []
            for _ in range(4):
                values, norm = update(current_batch)
                metrics.append({**values, "gradient_norm": norm})
            torch.cuda.synchronize()
            record.update({"passed": True, "samples_per_second": 4 * size / (time.perf_counter() - started),
                           "peak_memory_bytes": torch.cuda.max_memory_allocated(), "metrics": metrics})
        except torch.cuda.OutOfMemoryError:
            record.update({"passed": False, "reason": "CUDA OOM"})
            workspace.optimizer.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()
        benchmarks.append(record)
        print(json.dumps({"event": "batch_benchmark", **record}), flush=True)
    passed = [record for record in benchmarks if record["passed"]]
    if not passed:
        raise RuntimeError("No batch candidate passes H200 preflight")
    selected = max(passed, key=lambda record: record["samples_per_second"])
    report = {
        "passed": True, "source_sha256": source_fingerprint(),
        "data_sha256": manifest_fingerprint(dataset.manifest), "normalizer_sha256": normalizer_hash,
        "gpu": torch.cuda.get_device_name(0), "benchmarks": benchmarks,
        "selected_batch": selected["batch_size"], "lambda_vision": lambda_vision,
        "lambda_tactile": lambda_tactile, "gradient_audits": audits,
        "parameters": model.parameter_counts(), "task_windows": dataset.task_lengths,
        "checks": ["all_eight_mask_paths", "no_future_leak", "all_modules_receive_gradients",
                   "EMA_initialized_and_frozen", "100_step_diffusion", "exact_optimizer_rng_resume",
                   "normalizer_restore", "finite_training_steps"],
    }
    temporary = report_path.with_suffix('.partial')
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False))
    temporary.replace(report_path)
    del workspace, model, batch, tiny_batch, calibration_batch, current_batch
    gc.collect()
    torch.cuda.empty_cache()
    return report, dataset


def main() -> None:
    """Run bounded preflight; on success start a fresh, independent formal run."""
    import argparse
    import signal

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--preflight-only', action='store_true')
    arguments = parser.parse_args()
    cfg = OmegaConf.load(arguments.config)
    OmegaConf.resolve(cfg)
    cfg.output_dir = str(arguments.output_dir)
    report_path = arguments.output_dir / 'preflight' / 'report.json'
    dataset = None
    if report_path.exists():
        report = json.loads(report_path.read_text())
        if not report['passed'] or report['source_sha256'] != source_fingerprint():
            raise ValueError('Existing preflight report is not source-matched')
    else:
        report, dataset = run_preflight(cfg, report_path)
    if arguments.preflight_only:
        return
    cfg.dataloader.batch_size = report['selected_batch']
    cfg.runtime.preflight_report = str(report_path)
    latest = arguments.output_dir / 'checkpoints' / 'latest.ckpt'
    if latest.exists():
        cfg.training.resume = str(latest)
    workspace = Stage1V2Workspace(cfg, output_dir=str(arguments.output_dir))
    workspace.dataset = dataset
    for signum in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, lambda received, frame: workspace.request_stop(signal.Signals(received).name))
    workspace.run()


if __name__ == '__main__':
    main()
