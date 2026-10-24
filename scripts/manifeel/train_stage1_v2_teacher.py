"""Launch the standalone Stage 1 v2 Teacher workspace."""

from __future__ import annotations

import argparse
import signal
from pathlib import Path

import hydra
from omegaconf import OmegaConf


def main() -> None:
    """Resolve the v2 config and run only Stage 1 Teacher training."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "configs/manifeel/stage1_v2_teacher.yaml",
    )
    parser.add_argument("overrides", nargs="*", help="Strict OmegaConf key=value overrides")
    arguments = parser.parse_args()
    cfg = OmegaConf.load(arguments.config)
    OmegaConf.set_struct(cfg, True)
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(arguments.overrides))
    OmegaConf.resolve(cfg)
    workspace_class = hydra.utils.get_class(cfg._target_)
    workspace = workspace_class(cfg, output_dir=cfg.output_dir)
    for signum in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, lambda received, frame: workspace.request_stop(signal.Signals(received).name))
    workspace.run()


if __name__ == "__main__":
    main()
