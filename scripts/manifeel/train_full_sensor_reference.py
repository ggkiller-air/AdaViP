"""Run the independent Stage-0 workspace with explicit OmegaConf overrides."""

from __future__ import annotations

import argparse
import os
import signal
from datetime import timedelta
from pathlib import Path

import hydra
from omegaconf import OmegaConf


def main() -> None:
    """Resolve configuration before constructing the workspace and its policy."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=Path(__file__).resolve().parents[2] / "configs/manifeel/train_full_sensor_reference.yaml",
    )
    parser.add_argument("overrides", nargs="*", help="Strict key=value OmegaConf overrides")
    arguments = parser.parse_args()
    cfg = OmegaConf.load(arguments.config)
    OmegaConf.set_struct(cfg, True)
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(arguments.overrides))
    OmegaConf.resolve(cfg)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        import torch
        import torch.distributed as distributed

        from adavip.manifeel.distributed_full_sensor_workspace import DistributedFullSensorWorkspace

        if int(os.environ.get("LOCAL_WORLD_SIZE", world_size)) != world_size:
            raise ValueError("Only single-node DDP is supported for shared-cache training")
        if cfg.training.device != "cpu":
            torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        distributed.init_process_group(
            backend="gloo" if cfg.training.device == "cpu" else "nccl", timeout=timedelta(minutes=30),
        )
        workspace_class = DistributedFullSensorWorkspace
    else:
        workspace_class = hydra.utils.get_class(cfg._target_)
    workspace = workspace_class(cfg, output_dir=cfg.output_dir)

    def request_stop(signum: int, frame: object) -> None:
        workspace.request_stop(signal.Signals(signum).name)

    for signum in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, request_stop)
    workspace.run()
    if workspace.throughput_preflight_failed:
        raise RuntimeError("Throughput preflight failed; checkpoint saved instead of continuing inefficient training")
    if world_size > 1:
        distributed.destroy_process_group()


if __name__ == "__main__":
    main()
