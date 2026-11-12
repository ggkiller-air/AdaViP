"""Structural ablations, fixed labels, common weights and exact skipped-A continuation."""

import copy

import numpy as np
import pytest
import torch

from adavip.manifeel.full_sensor_evaluation import load_evaluation_policy
from adavip.manifeel.stage2_ablation_workspace import tensor_fingerprint, validate_gradient_routes, Stage2EpochSampler
from adavip.manifeel.stage2_data import task_utility_permutation, Stage2UtilityDataset
from adavip.manifeel.stage2_main_ablation import MainAblationTrainer, audit_pruned_initialization
from scripts.manifeel.train_stage2_t2 import set_stage_trainability
from test_stage2_l1 import policy, make_trainer
from test_stage2_policy import make_batch


def test_all_variants_start_with_identical_action_function():
    full = policy()
    batch = make_batch()
    sample = full.prepare_denoising_sample(batch)
    expected = full.training_losses(batch, "B", sample)
    for variant in "BCDEF":
        model = copy.deepcopy(full)
        model.apply_ablation(variant)
        actual = model.training_losses(batch, "B", sample)
        torch.testing.assert_close(actual["conditioning"], expected["conditioning"], rtol=0, atol=0)
        torch.testing.assert_close(actual["action_loss_per_sample"], expected["action_loss_per_sample"], rtol=0, atol=0)


@pytest.mark.parametrize("variant", list("BCDEF"))
def test_pruning_preserves_rng_weights_and_routes(variant):
    model = policy()
    original = copy.deepcopy(model.state_dict())
    rng = torch.get_rng_state().clone()
    model.apply_ablation(variant)
    audit_pruned_initialization(original, model)
    assert torch.equal(rng, torch.get_rng_state())
    assert (model.utility_predictor is None) == (variant in "BE")
    assert (model.film_hypernets is None) == (variant in "CE")
    assert (model.query_projection is None) == (variant in "BDE")
    batch = make_batch()
    state = tensor_fingerprint(model.state_dict())
    rng = torch.get_rng_state().clone()
    validate_gradient_routes(model, batch)
    assert state == tensor_fingerprint(model.state_dict())
    assert torch.equal(rng, torch.get_rng_state())
    set_stage_trainability(model, "C")
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.001)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        losses = model.training_losses(batch, "C")
        (losses["action_loss"] + .05 * losses["utility_loss"]).backward()
        optimizer.step()
    encoded = model.encode_stage2(batch["obs"])
    if variant in "CE":
        assert encoded["modulated"] is encoded["tokens"]
    if variant in "BDE":
        assert encoded["query"] is encoded["z_t"]
    if variant in "BE":
        del batch["utility"]
        assert model.training_losses(batch, "C")["utility_loss"] == 0


def test_shuffle_bijection_task_vectors_and_rng():
    tasks = np.repeat(np.arange(9), np.arange(11, 20))
    np.random.seed(42)
    before = np.random.get_state()
    rows = task_utility_permutation(tasks, 20261109)
    after = np.random.get_state()
    assert np.array_equal(before[1], after[1]) and before[2:] == after[2:]
    assert np.array_equal(rows, task_utility_permutation(tasks, 20261109))
    assert not np.array_equal(rows, task_utility_permutation(tasks, 42))
    assert np.array_equal(np.sort(rows), np.arange(len(tasks)))
    assert np.array_equal(tasks[rows], tasks)
    with pytest.raises(ValueError, match="restricted"):
        Stage2UtilityDataset.shuffle_utility(None, 123, split="validation")


def test_skipping_a_preserves_action_weights_random_stream_and_sampling(tmp_path):
    full = make_trainer(tmp_path / "full_a", stop=2)
    initial = copy.deepcopy(full.policy.state_dict())
    rng = torch.get_rng_state().clone()
    full.run()
    assert full.global_step == 2
    assert torch.equal(rng, torch.get_rng_state())
    for name, value in full.policy.state_dict().items():
        if not name.startswith("utility_predictor."):
            torch.testing.assert_close(value, initial[name], rtol=0, atol=0)
    for variant in "BE":
        other = trainer(tmp_path / variant, variant)
        for epoch in (2, 6, 7, 600):
            a, b = Stage2EpochSampler(full.dataset, 4), Stage2EpochSampler(other.dataset, 4)
            a.epoch = b.epoch = epoch
            assert list(a) == list(b)
        for step in (2040, 12239, 12240, 12739, 1223999, 1224000):
            from scripts.manifeel.train_stage2_t2 import make_optimizer_scheduler
            _, expected = make_optimizer_scheduler(full.policy, 2040)
            _, actual = make_optimizer_scheduler(other.policy, 2040)
            assert [fn(step) for fn in expected.lr_lambdas] == [fn(step) for fn in actual.lr_lambdas]


def trainer(output, variant, stop=0):
    original = make_trainer(output)
    model = original.policy
    model.apply_ablation(variant)
    cfg = original.cfg
    cfg["policy"]["ablation"] = variant
    contract = {**original.contract, "ablation": variant}
    return MainAblationTrainer(model, original.dataset, output, contract, cfg, lambda: 1,
                               device="cpu", batch_size=4, epochs=7, workers=0, stop_after_steps=stop)


@pytest.mark.parametrize("variant", list("BCDEF"))
def test_exact_resume_clock_and_portable_inference(tmp_path, variant):
    reference = make_trainer(tmp_path / "reference")
    full = trainer(tmp_path / "full", variant)
    skipped = 2 if variant in "BE" else 0
    assert full.global_step == skipped
    assert full.epoch == (2 if skipped else 1)
    for group, base, fn in zip(full.optimizer.param_groups, reference.scheduler.base_lrs,
                               reference.scheduler.lr_lambdas):
        assert group["lr"] == base * fn(skipped)
    assert not full.optimizer.state
    full.run()
    stopped = trainer(tmp_path / "resumed", variant, stop=3)
    stopped.run()
    resumed = trainer(stopped.output, variant)
    resumed.run(stopped.output / "checkpoints/latest.ckpt")
    assert full.global_step == resumed.global_step == 14
    assert sum(full.cumulative_counts) == (14 - skipped) * 4
    for role in ("policy", "ema"):
        for name, value in getattr(full, role).state_dict().items():
            torch.testing.assert_close(value, getattr(resumed, role).state_dict()[name], rtol=0, atol=0)
    for index, state in full.optimizer.state_dict()["state"].items():
        for key, value in state.items():
            torch.testing.assert_close(value, resumed.optimizer.state_dict()["state"][index][key], rtol=0, atol=0)
    assert full.scheduler.state_dict() == resumed.scheduler.state_dict()
    for epoch in (1, 6, 7):
        assert (full.output / f"checkpoints/epoch{epoch}.ckpt").is_file()
    restored, _, _ = load_evaluation_policy(full.output / "checkpoints/latest.ckpt")
    obs = make_batch()["obs"]
    with torch.no_grad():
        a = restored.predict_action(obs, generator=torch.Generator().manual_seed(9))
        b = full.ema.predict_action(obs, generator=torch.Generator().manual_seed(9))
    torch.testing.assert_close(a["action"], b["action"], rtol=0, atol=0)
