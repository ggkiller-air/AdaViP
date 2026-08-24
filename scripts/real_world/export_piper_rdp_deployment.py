#!/usr/bin/env python3
"""Export the Piper RDP EMA policy and preprocessing as a deployment bundle."""

from __future__ import annotations

import argparse
import datetime
import gc
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import tarfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

import dill
import torch
from omegaconf import OmegaConf

from adavip.real_world.gelsight_marker_processor import MarkerDetectorConfig
from adavip.real_world.piper_rdp_runtime import ARTIFACT_FORMAT


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAINING_OUTPUT = Path(
    os.environ.get("ADAVIP_RDP_OUTPUT", REPO_ROOT / "outputs" / "piper_rdp")
)
DEFAULT_CHECKPOINT = DEFAULT_TRAINING_OUTPUT / "ldp/checkpoints/latest.ckpt"
DEFAULT_DATASET = Path(
    os.environ.get("ADAVIP_RDP_DATASET", REPO_ROOT / "datasets" / "piper_rdp")
)
DEFAULT_OUTPUT = DEFAULT_TRAINING_OUTPUT / "deployment/piper_rdp_18_75hz_ema"
DEFAULT_SAMPLE_INDICES = "20,127,327,404,550,654"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--num-inference-steps", type=int, default=8)
    parser.add_argument("--sample-indices", default=DEFAULT_SAMPLE_INDICES)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cpu-threads", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolved_container(config: Any) -> Any:
    """Convert an OmegaConf node to plain resolved values."""
    return OmegaConf.to_container(config, resolve=True, enum_to_str=True)


def sanitize_policy_config(cfg: Any, num_inference_steps: int) -> dict[str, Any]:
    """Remove external AT checkpoint and CUDA requirements from policy config."""
    policy_config = resolved_container(cfg.policy)
    policy_config["at"]["load_dir"] = None
    policy_config["at"]["device"] = "cpu"
    policy_config["num_inference_steps"] = num_inference_steps
    return policy_config


def sanitize_dataset_config(cfg: Any) -> dict[str, Any]:
    """Use the sampling dataset without recomputing latent normalization."""
    dataset_config = resolved_container(cfg.task.dataset)
    dataset_config["_target_"] = (
        "adavip.real_world.piper_rdp_dataset.PiperRdpDataset"
    )
    dataset_config.pop("at", None)
    dataset_config.pop("use_latent_action_before_vq", None)
    return dataset_config


def build_contract(cfg: Any) -> dict[str, Any]:
    """Describe model tensors and the reactive token-decoding protocol."""
    shape_meta = resolved_container(cfg.task.shape_meta)
    action_dim = int(shape_meta["action"]["shape"][0])
    latent_feature_dim = int(cfg.policy.at.n_embed)
    latent_time_steps = 8
    return {
        "control_frequency_hz": 18.75,
        "action_semantics": "absolute Piper joint_1..joint_6 and gripper target",
        "action_dim": action_dim,
        "horizon": int(cfg.horizon),
        "action_steps": int(cfg.n_action_steps),
        "dataset_observation_steps": int(cfg.dataset_obs_steps),
        "slow_observation_steps": int(cfg.n_obs_steps),
        "slow_observation_stride_frames": int(
            cfg.dataset_obs_temporal_downsample_ratio
        ),
        "slow_observations": {
            key: list(attributes["shape"])
            for key, attributes in shape_meta["obs"].items()
        },
        "extended_observations": {
            key: list(attributes["shape"])
            for key, attributes in shape_meta["extended_obs"].items()
        },
        "latent_action": {
            "encoded_shape": [latent_time_steps, latent_feature_dim],
            "flattened_token_shape": [latent_time_steps * latent_feature_dim],
            "policy_token_schedule_shape": [
                int(cfg.n_action_steps),
                latent_time_steps * latent_feature_dim,
            ],
        },
        "runtime_protocol": [
            "Call predict_action(..., return_latent_action=True) on slow observations.",
            "Take one 64D row from result['action']; all scheduled rows contain the same token.",
            "At each control step collect the available 15D tactile PCA history.",
            "Call predict_from_latent_action(token, tactile_history, history_length, 2).",
            "Execute the last returned absolute 7D action.",
        ],
        "image": {"color": "RGB", "dtype": "float32", "range": [0.0, 1.0]},
    }


def package_versions() -> dict[str, str]:
    """Return the deployment environment versions available at export time."""
    distributions = (
        "torch",
        "torchvision",
        "numpy",
        "scipy",
        "hydra-core",
        "omegaconf",
        "diffusers",
        "einops",
    )
    return {
        name: importlib.metadata.version(name)
        for name in distributions
        if importlib.util.find_spec(name.replace("-core", "")) is not None
    }


def copy_runtime_sources(destination: Path) -> None:
    """Copy the local runtime and pinned upstream policy implementation."""
    adavip_output = destination / "adavip"
    real_world_output = adavip_output / "real_world"
    real_world_output.mkdir(parents=True)
    (adavip_output / "__init__.py").write_text(
        '"""Minimal Piper RDP deployment runtime."""\n', encoding="utf-8"
    )
    for relative_path in (
        Path("adavip/real_world/__init__.py"),
        Path("adavip/real_world/gelsight_marker_processor.py"),
        Path("adavip/real_world/piper_rdp_runtime.py"),
    ):
        shutil.copy2(REPO_ROOT / relative_path, destination / relative_path)

    upstream_root = REPO_ROOT / "third_party/reactive_diffusion_policy"
    shutil.copytree(
        upstream_root / "reactive_diffusion_policy",
        destination / "reactive_diffusion_policy",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    license_output = destination / "THIRD_PARTY_LICENSES"
    license_output.mkdir()
    shutil.copy2(
        upstream_root / "LICENSE",
        license_output / "reactive_diffusion_policy_LICENSE",
    )


def prepare_bundle(
    checkpoint: Path,
    dataset: Path,
    output: Path,
    num_inference_steps: int,
    overwrite: bool,
) -> Path:
    """Write an unarchived self-contained bundle ready for offline checking."""
    checkpoint = checkpoint.resolve()
    dataset = dataset.resolve()
    output = output.resolve()
    archive = output.with_suffix(".tar.gz")
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if num_inference_steps < 1:
        raise ValueError("num_inference_steps must be positive")
    if (output.exists() or archive.exists()) and not overwrite:
        raise FileExistsError(f"Deployment output already exists: {output}")

    with checkpoint.open("rb") as stream:
        payload = torch.load(stream, map_location="cpu", pickle_module=dill)
    cfg = payload["cfg"]
    if not bool(cfg.training.use_ema) or "ema_model" not in payload["state_dicts"]:
        raise RuntimeError("Expected an EMA policy in the LDP checkpoint")
    policy_state = payload["state_dicts"]["ema_model"]
    if not any(key.startswith("normalizer.") for key in policy_state):
        raise RuntimeError("EMA state dict does not contain normalizers")

    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    try:
        policy_artifact = temporary / "piper_rdp_ema.pt"
        artifact = {
            "format": ARTIFACT_FORMAT,
            "created_at_utc": datetime.datetime.now(
                datetime.timezone.utc
            ).isoformat(),
            "weights": "ema_model",
            "source_checkpoint": str(checkpoint),
            "source_checkpoint_sha256": sha256_file(checkpoint),
            "policy_config": sanitize_policy_config(cfg, num_inference_steps),
            "policy_state_dict": policy_state,
            "offline_dataset_config": sanitize_dataset_config(cfg),
            "contract": build_contract(cfg),
            "runtime": {
                "num_inference_steps": num_inference_steps,
                "warmup_required": True,
                "dataset_obs_temporal_downsample_ratio": int(
                    cfg.dataset_obs_temporal_downsample_ratio
                ),
            },
            "torch_version": torch.__version__,
        }
        torch.save(artifact, policy_artifact)

        del payload, policy_state, artifact
        gc.collect()

        pca_output = temporary / "gelsight_pca"
        pca_output.mkdir()
        pca_source = dataset / "artifacts" / "gelsight_pca"
        for name in ("pca_transform_matrix.npy", "pca_mean_matrix.npy"):
            shutil.copy2(pca_source / name, pca_output / name)
        copy_runtime_sources(temporary)

        versions = package_versions()
        (temporary / "requirements.txt").write_text(
            "".join(f"{name}=={version}\n" for name, version in versions.items()),
            encoding="utf-8",
        )

        manifest = {
            "format": "adavip.piper_rdp.deployment_bundle.v1",
            "policy_artifact": policy_artifact.name,
            "policy_artifact_bytes": policy_artifact.stat().st_size,
            "policy_artifact_sha256": sha256_file(policy_artifact),
            "source_checkpoint": str(checkpoint),
            "weights": "ema_model",
            "detector": asdict(MarkerDetectorConfig()),
            "pca_transform": "gelsight_pca/pca_transform_matrix.npy",
            "pca_mean": "gelsight_pca/pca_mean_matrix.npy",
            "offline_inference_report": "offline_inference_report.json",
            "environment": {
                "python": sys.version.split()[0],
                "packages": versions,
                "reactive_diffusion_policy_revision": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"],
                    cwd=REPO_ROOT / "third_party/reactive_diffusion_policy",
                    text=True,
                ).strip(),
            },
            "contract": build_contract(cfg),
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (temporary / "README.txt").write_text(
            "Piper RDP 18.75 Hz EMA deployment bundle\n"
            "Add this directory to PYTHONPATH, then load piper_rdp_ema.pt with "
            "adavip.real_world.piper_rdp_runtime.load_policy_artifact().\n"
            "Initialize OnlineGelSightPcaProcessor with the first no-contact "
            "GelSight frame of every episode.\n"
            "Use RGB float32 images in [0, 1] with CHW layout; the policy's eval "
            "mode applies the trained center crop.\n"
            "See manifest.json for tensor shapes, dependencies, checksums, and "
            "the reactive decoding protocol.\n",
            encoding="utf-8",
        )
        if output.exists():
            shutil.rmtree(output)
        os.replace(temporary, output)
        return output
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def finalize_bundle(output: Path) -> tuple[Path, dict[str, Any]]:
    """Add audited file hashes and create the final gzip archive."""
    output = output.resolve()
    report = output / "offline_inference_report.json"
    if not report.is_file():
        raise FileNotFoundError(f"Offline inference report is missing: {report}")
    report_payload = json.loads(report.read_text(encoding="utf-8"))
    if not report_payload.get("passed"):
        raise RuntimeError("Offline inference report did not pass")

    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = {}
    for path in sorted(output.rglob("*")):
        if path.is_file() and path != manifest_path:
            relative_path = path.relative_to(output).as_posix()
            files[relative_path] = {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
    manifest["files"] = files
    manifest["offline_inference_passed"] = True
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    archive = output.with_suffix(".tar.gz")
    temporary_archive = archive.with_name(f".{archive.name}.{os.getpid()}.tmp")
    try:
        with tarfile.open(temporary_archive, "w:gz", compresslevel=1) as tar:
            tar.add(output, arcname=output.name)
        os.replace(temporary_archive, archive)
    finally:
        temporary_archive.unlink(missing_ok=True)
    sizes = {
        "bundle_directory_bytes": sum(
            path.stat().st_size for path in output.rglob("*") if path.is_file()
        ),
        "archive_bytes": archive.stat().st_size,
    }
    return archive, sizes


def main() -> int:
    args = parse_args()
    output = prepare_bundle(
        args.checkpoint,
        args.dataset,
        args.output,
        args.num_inference_steps,
        args.overwrite,
    )
    inference_script = REPO_ROOT / "scripts/real_world/infer_piper_rdp_offline.py"
    subprocess.run(
        [
            sys.executable,
            str(inference_script),
            "--bundle",
            str(output),
            "--dataset",
            str(args.dataset.resolve()),
            "--sample-indices",
            args.sample_indices,
            "--device",
            args.device,
            "--cpu-threads",
            str(args.cpu_threads),
        ],
        cwd=REPO_ROOT,
        check=True,
    )
    archive, sizes = finalize_bundle(output)
    print(f"bundle={output}")
    print(f"bundle_bytes={sizes['bundle_directory_bytes']}")
    print(f"archive={archive}")
    print(f"archive_bytes={sizes['archive_bytes']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
