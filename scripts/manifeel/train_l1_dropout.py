"""Train the L1 10% input-dropout ablation from scratch on one allocated H200."""

from __future__ import annotations

import argparse
import json
import os
import signal
from pathlib import Path

import torch
from omegaconf import OmegaConf

from adavip.manifeel.l1_dropout import L1DropoutWorkspace
from adavip.manifeel.full_sensor_schema import IMAGE_KEYS
from torch.utils.data import default_collate
from diffusion_policy.common.pytorch_util import dict_apply
from scripts.manifeel.language_runtime import configure_profile


def preflight(workspace: L1DropoutWorkspace) -> None:
    """Check a real batch32 and all masks without changing weights, optimizer or RNG."""
    with torch.random.fork_rng(devices=[0]):
        dataset = workspace._make_dataset()
        policy = workspace.model
        # Loading this normalizer rebuilds its ParameterDict on CPU.
        policy.set_normalizer(dataset.get_normalizer())
        policy.to("cuda:0")
        if any(parameter.device != policy.device for parameter in policy.normalizer.parameters()):
            raise RuntimeError("Preflight normalizer must share the policy device")
        batch = dict_apply(default_collate([dataset[index] for index in range(32)]),
                           lambda tensor: tensor.to("cuda:0"))
        encoder = policy.obs_encoder
        observations = policy.normalizer.normalize(batch["obs"])
        if any(value.device != policy.device for value in observations.values()):
            raise RuntimeError("Preflight normalized observations must stay on the policy device")
        with torch.no_grad():
            tokens = encoder.encode_sensor_tokens(observations)
            language = encoder.language_embedding(observations)
            expected = encoder.condition_state(encoder.fuse_state_tokens(tokens), language)
            actual = encoder.condition_state(encoder.fuse_state_tokens(
                tokens, torch.ones(32, 6, dtype=torch.bool, device=policy.device)), language)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        ids = torch.arange(32, device=policy.device) % 10
        sample = policy.prepare_denoising_sample(batch, generator=torch.Generator(device=policy.device).manual_seed(700042))
        losses = policy.compute_loss_per_sample(batch, ids, sample)
        losses.mean().backward()
        gradient_norms = {key: float(next(encoder.image_encoders[key].parameters()).grad.norm())
                          for key in IMAGE_KEYS}
        if not torch.isfinite(losses).all() or not all(value > 0 for value in gradient_norms.values()):
            raise RuntimeError("L1 dropout real-data preflight failed")
        norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), workspace.cfg.training.gradient_clip_norm,
                                             error_if_nonfinite=True)
        policy.zero_grad(set_to_none=True)
        report = {"passed": True, "batch_size": 32, "all_ten_conditions": True,
                  "full_input_exact_parity": True, "action_loss": float(losses.mean()),
                  "gradient_norm": float(norm), "image_gradient_norms": gradient_norms,
                  "optimizer_updates": 0, "source_fingerprint": workspace._source_fingerprint(),
                  "normalizer_device": str(next(policy.normalizer.parameters()).device),
                  "normalized_observation_device": str(observations["state"].device),
                  "trainer_affinity": sorted(os.sched_getaffinity(0))}
        (Path(workspace.output_dir) / "preflight.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"event": "preflight_passed", **report}), flush=True)


def main() -> None:
    """Enforce batch32/E100 and save exact continuation state on scheduler signals."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    OmegaConf.set_struct(cfg, True)
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.overrides))
    OmegaConf.resolve(cfg)
    if (cfg.variant != "l1" or cfg.dataloader.batch_size != 32 or cfg.training.num_epochs != 100
            or cfg.policy.full_input_probability != 0.9 or cfg.optimizer.lr != 8e-5
            or cfg.training.seed != 42 or cfg.training.lr_warmup_steps != 500
            or list(cfg.training.checkpoint_epochs) != [20, 40, 60, 80, 100]):
        raise ValueError("Expected L1/drop10/batch32/E100/LR8e-5/seed42/checkpoints every20 epochs")
    if (not torch.cuda.is_available() or torch.cuda.device_count() != 1
            or "H200" not in torch.cuda.get_device_name(0)):
        raise RuntimeError("Training requires exactly one allocated H200")
    profile = os.environ.get("LC_RUNTIME_PROFILE", "stage0")
    runtime = {"profile": profile, **configure_profile(profile),
               "job_id": os.environ.get("SLURM_JOB_ID"),
               "trainer_affinity": sorted(os.sched_getaffinity(0)),
               "allocated_cpus": os.environ.get("SLURM_CPUS_PER_TASK")}
    output = Path(cfg.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / f"runtime_{os.environ.get('SLURM_JOB_ID', 'local')}.json").write_text(
        json.dumps(runtime, indent=2) + "\n")
    print(json.dumps({"event": "l1_dropout_runtime", **runtime}), flush=True)
    workspace = L1DropoutWorkspace(cfg, cfg.output_dir)

    def request_stop(signum: int, frame: object) -> None:
        workspace.request_stop(signal.Signals(signum).name)

    for signum in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, request_stop)
    if not cfg.resume:
        preflight(workspace)
    workspace.run()


if __name__ == "__main__":
    main()
