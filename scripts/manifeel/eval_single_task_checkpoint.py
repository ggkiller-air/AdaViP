#!/usr/bin/env python3
"""Evaluate one ManiFeel checkpoint on an explicit physical GPU."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import gc
import json
import os
from pathlib import Path
import sys

import isaacgym  # noqa: F401  # Isaac Gym must be imported before torch.
import dill
import hydra
from hydra.core.global_hydra import GlobalHydra
import numpy as np
from omegaconf import OmegaConf
import torch
import wandb

from adavip.manifeel.eval_config import configure_single_task_evaluation
from diffusion_policy.workspace.base_workspace import BaseWorkspace


OmegaConf.register_new_resolver("eval", eval, replace=True)
REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEEL_CONFIG_ROOT = REPO_ROOT / "third_party" / "manifeel" / "manifeel" / "config"


def parse_args() -> argparse.Namespace:
    """Parse single-task evaluation arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--n-test", type=int, default=10)
    parser.add_argument("--n-test-vis", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--test-start-seed", type=int, default=100000)
    parser.add_argument("--diagnostic", action="store_true")
    parser.add_argument("--attention-trace", action="store_true",
                        help="record per-head fusion weights and rollout diagnostics")
    parser.add_argument("--hypernet-trace", action="store_true",
                        help="record HyperNet modulation, fusion attention and physical rollout state")
    parser.add_argument("--hypernet-intervention", choices=("none", "mean", "zero"), default="none")
    parser.add_argument("--modulation-trace", type=Path,
                        help="independent demonstration calibration trace for fixed mean modulation")
    parser.add_argument("--hypernet-context-intervention", choices=("none", "vision", "tactile", "state"), default="none")
    parser.add_argument("--context-calibration-trace", type=Path,
                        help="independent demo feature trace for freezing one HyperNet context slice")
    parser.add_argument("--intervention-phase", choices=("all", "pre_contact", "post_contact_pre_success"), default="all")
    parser.add_argument("--ablation", choices=("none", "no_film", "no_attention", "no_tactile"),
                        default="none")
    parser.add_argument("--canonicalize-quaternion", action="store_true")
    return parser.parse_args()


def json_value(value: object) -> object:
    """Convert NumPy scalar values to JSON-compatible Python scalars."""
    if isinstance(value, np.generic):
        return value.item()
    return value


def main() -> None:
    """Load the saved Hydra config and run its task on one physical GPU."""
    args = parse_args()
    if args.hypernet_intervention != "none" and args.hypernet_context_intervention != "none":
        raise ValueError("output and context interventions must be evaluated separately")
    if args.intervention_phase != "all" and args.hypernet_intervention == args.hypernet_context_intervention == "none":
        raise ValueError("phase selection requires a modulation or context intervention")
    if not args.device.startswith("cuda:"):
        raise ValueError(f"device must be cuda:<physical-index>, got {args.device!r}")
    if args.output_dir.exists():
        raise FileExistsError(f"Evaluation output already exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True)

    with args.checkpoint.open("rb") as checkpoint_file:
        payload = torch.load(
            checkpoint_file,
            pickle_module=dill,
            map_location="cpu",
        )
    if "cfg" not in payload:
        raise KeyError(f"Checkpoint has no saved Hydra config: {args.checkpoint}")
    if not GlobalHydra.instance().is_initialized():
        hydra.initialize_config_dir(
            config_dir=str(MANIFEEL_CONFIG_ROOT),
            version_base=None,
        )

    cfg = payload["cfg"]
    configure_single_task_evaluation(
        cfg,
        device=args.device,
        n_test=args.n_test,
        n_test_vis=args.n_test_vis,
        max_steps=args.max_steps,
        test_start_seed=args.test_start_seed,
    )

    workspace_class = hydra.utils.get_class(cfg._target_)
    workspace = workspace_class(cfg, output_dir=str(args.output_dir))
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    policy = workspace.ema_model if cfg.training.use_ema else workspace.model
    policy.to(torch.device(args.device))
    policy.eval()
    del payload, workspace
    gc.collect()
    intervention_stack = ExitStack()
    replacement = None
    if args.hypernet_intervention != "none":
        from adavip.manifeel.action_probe import override_hypernet

        if not hasattr(policy.obs_encoder, "hypernet"):
            raise ValueError("modulation interventions require a HyperResNet checkpoint")
        if args.hypernet_intervention == "mean":
            if args.modulation_trace is None:
                raise ValueError("mean modulation requires --modulation-trace")
            with np.load(args.modulation_trace, allow_pickle=False) as trace:
                bounded = trace["calibration_mean_gamma_beta"]
            replacement = torch.from_numpy(np.arctanh(np.clip(bounded, -1 + 1e-6, 1 - 1e-6))).to(policy.device)
        else:
            replacement = torch.zeros(1, policy.obs_encoder.hypernet[-1].out_features, device=policy.device)

    runner = hydra.utils.instantiate(
        cfg.task.env_runner,
        output_dir=str(args.output_dir),
    )
    rollout_recorder = None
    policy_recorder = None
    if args.diagnostic or args.attention_trace or args.hypernet_trace or args.hypernet_intervention != "none" or args.hypernet_context_intervention != "none" or args.ablation != "none" or args.canonicalize_quaternion:
        from adavip.manifeel.diagnostics import attach_diagnostics

        rollout_recorder, policy_recorder = attach_diagnostics(
            runner, policy, args.output_dir, ablation=args.ablation,
            canonicalize_quaternion=args.canonicalize_quaternion)
    phase_selections = []
    environment_selector = None
    if args.intervention_phase != "all":
        def select_phase() -> np.ndarray:
            selected = rollout_recorder.current_contact_phase(args.intervention_phase)
            phase_selections.append(selected.copy())
            return selected

        environment_selector = select_phase
    if replacement is not None:
        intervention_stack.enter_context(override_hypernet(policy.obs_encoder, replacement,
            environment_selector=environment_selector, n_obs_steps=policy.n_obs_steps))
    if args.hypernet_context_intervention != "none":
        from adavip.manifeel.action_probe import override_hypernet_context

        if not hasattr(policy.obs_encoder, "hypernet") or args.context_calibration_trace is None:
            raise ValueError("context interventions require HyperResNet and a separate demonstration calibration trace")
        with np.load(args.context_calibration_trace, allow_pickle=False) as trace:
            features = torch.from_numpy(trace["base_features"])
            state = torch.from_numpy(trace["normalized_state"])
        if features.ndim != 3 or state.ndim != 2:
            raise ValueError("context calibration requires independent demonstration frames, not rollout histories")
        width = features.shape[-1]
        reference = torch.cat((torch.nn.functional.layer_norm(features[:, 0], (width,)),
                               torch.nn.functional.layer_norm(features[:, 1], (width,)), state), dim=-1).mean(dim=0, keepdim=True)
        intervention_stack.enter_context(override_hypernet_context(policy.obs_encoder, reference,
            args.hypernet_context_intervention, environment_selector=environment_selector, n_obs_steps=policy.n_obs_steps))
    attention_recorder = None
    if args.attention_trace and not args.hypernet_trace:
        from adavip.manifeel.attention_trace import AttentionTraceRecorder

        attention_recorder = AttentionTraceRecorder(policy, args.output_dir)
    hypernet_recorder = None
    if args.hypernet_trace:
        from adavip.manifeel.hypernet_probe import HyperNetRolloutRecorder

        hypernet_recorder = HyperNetRolloutRecorder(policy, args.output_dir)
    try:
        runner_log = runner.run(policy)
        runner_log["diagnostics/hypernet_intervention"] = args.hypernet_intervention
        runner_log["diagnostics/hypernet_context_intervention"] = args.hypernet_context_intervention
        runner_log["diagnostics/intervention_phase"] = args.intervention_phase
        if phase_selections:
            np.savez_compressed(args.output_dir / "intervention_phase_masks.npz", selected=np.stack(phase_selections))
            runner_log["diagnostics/intervened_environments_by_chunk"] = [int(selected.sum()) for selected in phase_selections]
        if hypernet_recorder is not None:
            runner_log["diagnostics/hypernet"] = hypernet_recorder.save()
        if attention_recorder is not None:
            runner_log["diagnostics/attention"] = attention_recorder.save()
        if rollout_recorder is not None and policy_recorder is not None:
            rollout_summary = rollout_recorder.save()
            policy_summary = policy_recorder.save()
            runner_log["diagnostics/rollout"] = rollout_summary
            runner_log["diagnostics/policy"] = policy_summary
            if not np.isclose(rollout_summary["success_rate"], runner_log["test/success_rate"]):
                raise ValueError("recorded per-environment success disagrees with runner success rate")
    finally:
        if hypernet_recorder is not None:
            hypernet_recorder.close()
        if attention_recorder is not None:
            attention_recorder.close()
        intervention_stack.close()
        runner.env.close()

    json_log = {
        key: value._path if isinstance(value, wandb.Video) else json_value(value)
        for key, value in runner_log.items()
    }
    output_path = args.output_dir / "eval_log.json"
    output_path.write_text(json.dumps(json_log, indent=2, sort_keys=True) + "\n")
    print(json.dumps(json_log, indent=2, sort_keys=True))
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
