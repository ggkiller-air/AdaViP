"""Native query preservation, identity initialization and UVIP gradient boundaries."""

import pytest

torch = pytest.importorskip("torch")
from adavip.ftp1.modulation import FTP1UtilityModulation, GROUPS, WIDTHS, per_sample_flow_loss


def fixture():
    torch.manual_seed(42)
    module = FTP1UtilityModulation()
    tokens = {name: torch.randn(2, 3, width, requires_grad=True)
              for name, width in zip(GROUPS, WIDTHS)}
    masks = {name: torch.ones(2, 3, dtype=torch.bool) for name in GROUPS}
    masks["T1"][0, 2] = False
    return module, tokens, masks


def test_identity_and_native_widths():
    module, tokens, masks = fixture()
    result = module(tokens, masks)
    assert result["utility"].shape == (2, 8)
    assert result["context"].shape == (2, 256)
    for name in GROUPS:
        torch.testing.assert_close(result["tokens"][name],
                                   tokens[name].masked_fill(~masks[name][..., None], 0), rtol=0, atol=0)
    assert not any(isinstance(layer, torch.nn.MultiheadAttention) for layer in module.modules())
    assert not any("query" in name or "action" in name for name, _ in module.named_parameters())


def test_utility_loss_only_updates_predictor():
    module, tokens, masks = fixture()
    module(tokens, masks)["utility"].square().mean().backward()
    assert all(value.grad is None for value in tokens.values())
    for name, param in module.named_parameters():
        assert (param.grad is not None) == name.startswith("utility_predictor.")


def test_action_gradient_cannot_update_predictor_and_missing_slots_stay_zero():
    module, tokens, masks = fixture()
    with torch.no_grad():
        for head in module.film.values():
            head.weight.normal_(std=0.01)
            head.bias.fill_(0.1)
    masks["T1"][0] = False
    result = module(tokens, masks)
    assert torch.count_nonzero(result["tokens"]["T1"][0]) == 0
    sum(value.square().mean() for value in result["tokens"].values()).backward()
    assert all(param.grad is None for param in module.utility_predictor.parameters())
    assert all(param.grad is not None for param in module.context.parameters())
    assert tokens["V1"].grad.abs().sum() > 0


def test_per_sample_loss_uses_native_action_and_time_mask():
    prediction = torch.zeros(2, 32, 120)
    target = torch.ones_like(prediction)
    prediction[0, :, 9:17] = 3
    prediction[:, :, 70:] = 1000
    mask = torch.zeros_like(prediction, dtype=torch.bool)
    mask[0, :, 9:17] = True
    mask[1, :20, 9:17] = True
    torch.testing.assert_close(per_sample_flow_loss(prediction, target, mask), torch.tensor([4., 1.]))
