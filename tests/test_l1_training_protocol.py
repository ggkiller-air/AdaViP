"""Prevent shared optimizer, scheduler, data and checkpoint settings from drifting."""

from pathlib import Path

import pytest

pytest.importorskip("omegaconf")
from omegaconf import OmegaConf

from scripts.manifeel.l1_training_protocol import validate_l1_training_protocol


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("filename", ["train_l1_dropout.yaml", "train_stage1_future.yaml"])
def test_shared_protocol_matches_original(filename: str) -> None:
    cfg = OmegaConf.load(ROOT / "configs/manifeel" / filename)
    assert validate_l1_training_protocol(cfg)["matched"]


@pytest.mark.parametrize("override", ["dataloader.batch_size=32", "optimizer.lr=0.00004",
                                     "training.num_epochs=100", "training.lr_warmup_steps=1000",
                                     "training.checkpoint_epochs=[20,40,60,80,100]",
                                     "ema.max_value=0.99", "dataset.seed=43",
                                     "policy.num_inference_steps=50"])
def test_reject_shared_protocol_drift(override: str) -> None:
    cfg = OmegaConf.load(ROOT / "configs/manifeel/train_stage1_future.yaml")
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist([override]))
    with pytest.raises(ValueError, match="Original L1 protocol mismatch"):
        validate_l1_training_protocol(cfg)
