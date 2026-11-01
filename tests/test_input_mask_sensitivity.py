"""Check paired diagnostic correctness and reuse of the exact training intervention."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("zarr")

from adavip.manifeel.full_sensor_schema import condition_presence
from adavip.manifeel.input_mask_sensitivity import fixed_noise, mask_conditionings, paired_mask_losses, summarize_losses
from adavip.manifeel.l1_dropout import L1DropoutWorkspace
from adavip.manifeel.language_conditioning_dataset import LanguageConditioningDataset
from adavip.manifeel.language_conditioning_workspace import LanguageConditioningWorkspace
from test_l1_dropout import dropout_config
from test_language_conditioning import data_args, small_config


def test_original_and_dropout_diagnostic_use_identical_masks(data_args: dict, tmp_path: Path) -> None:
    original = LanguageConditioningWorkspace(small_config(data_args, tmp_path / "original")).model
    dropout = L1DropoutWorkspace(dropout_config(data_args, tmp_path / "dropout")).model
    dataset = LanguageConditioningDataset(**data_args)
    for policy in (original, dropout):
        policy.set_normalizer(dataset.get_normalizer())
        policy.eval()
    batch = torch.utils.data.default_collate([dataset[3], dataset[36]])
    conditions = mask_conditionings(original, batch["obs"])
    torch.testing.assert_close(conditions, mask_conditionings(dropout, batch["obs"]), rtol=0, atol=0)
    for index, flags in enumerate(torch.from_numpy(condition_presence())):
        expected = dropout.obs_encoder(dropout.normalizer.normalize(batch["obs"]), flags.expand(2, -1))
        torch.testing.assert_close(conditions[index], expected, rtol=0, atol=0)
    values, timesteps = paired_mask_losses(original, batch, [3, 36], repeats=2)
    # Direct policy evaluation receives the same noise/steps under all ten masks.
    for repeat in range(2):
        noise, steps = fixed_noise([3, 36], repeat, original.horizon, 10)
        sample = dropout.prepare_denoising_sample(batch, noise=noise, timesteps=steps)
        for condition in range(10):
            actual = dropout.compute_loss_per_sample(batch, torch.full((2,), condition, dtype=torch.long), sample)
            np.testing.assert_array_equal(values[:, repeat, condition], actual.detach().numpy())
        np.testing.assert_array_equal(timesteps[:, repeat], steps.numpy())


def test_noise_independent_of_batch_boundaries_and_repeat() -> None:
    combined = fixed_noise([7, 12], 2, 16, 100)
    singles = [fixed_noise([index], 2, 16, 100) for index in (7, 12)]
    for array in range(2):
        torch.testing.assert_close(combined[array], torch.cat([value[array] for value in singles]), rtol=0, atol=0)
    assert not torch.equal(combined[0], fixed_noise([7, 12], 3, 16, 100)[0])


def test_signed_summaries_keep_harmful_and_helpful_effects() -> None:
    values = np.ones((3, 4, 10), dtype=np.float32)
    values[:, :, 1] += 2
    values[:, :, 2] -= 0.5
    result = summarize_losses(values)
    assert result["mean_delta"][:3] == [2, -0.5, 0]
    assert result["positive_fraction"][:3] == [1, 0, 0]
    assert result["zero_fraction"][:3] == [0, 0, 1]
    assert result["all_repeat_sign_agreement_fraction"][:3] == [1, 1, 0]
