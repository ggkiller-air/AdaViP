"""Teacher function retention, RNG neutrality, stage routing and strict continuation."""

import copy

import pytest
import torch

from adavip.manifeel.reference_diffusion_policy import FullSensorReferencePolicy
from adavip.manifeel.teacher_residual_validation import validate_teacher_parity, validate_residual_routes
from adavip.manifeel.stage2_ablation_workspace import tensor_fingerprint
from scripts.manifeel.train_stage2_t2 import set_stage_trainability, make_optimizer_scheduler
from test_stage2_l1 import policy
from test_stage2_policy import make_batch
from test_stage2_main_ablation import test_exact_resume_clock_and_portable_inference as check_resume


def test_teacher_residual_common_c_rng_and_exact_action():
    model = policy()
    canonical = copy.deepcopy(model)
    canonical.apply_ablation("C")
    rng = torch.get_rng_state().clone()
    model.apply_ablation("C_teacher_residual")
    assert torch.equal(rng, torch.get_rng_state())
    assert tensor_fingerprint(canonical.state_dict()) == tensor_fingerprint(
        {k: v for k, v in model.state_dict().items() if not k.startswith("w_out.")})
    teacher = FullSensorReferencePolicy(copy.deepcopy(model.obs_encoder), copy.deepcopy(model.noise_scheduler),
        horizon=16, n_obs_steps=2, n_action_steps=8, num_inference_steps=8,
        down_dims=(16, 32), diffusion_step_embed_dim=16, n_groups=8)
    teacher.set_normalizer(model.normalizer)
    assert tensor_fingerprint(teacher.model.state_dict()) != tensor_fingerprint(model.model.state_dict())
    model.initialize_from_teacher(teacher)
    batch = make_batch()
    assert validate_teacher_parity(teacher, model, batch, "cpu")["full_predict_action_exact"]
    state, rng = tensor_fingerprint(model.state_dict()), torch.get_rng_state().clone()
    report = validate_residual_routes(model, batch)
    assert state == tensor_fingerprint(model.state_dict())
    assert torch.equal(rng, torch.get_rng_state())
    assert report["B"]["initial"]["w_out"]["gradient_l2"] > 0
    assert report["B"]["initial"]["cross_attention"]["gradient_l2"] == 0
    assert report["B"]["after_w_out_update"]["cross_attention"]["gradient_l2"] > 0
    optimizer, _ = make_optimizer_scheduler(model, 2, epochs=7)
    action = next(g for g in optimizer.param_groups if g["name"] == "action")
    assert all(any(p is q for q in action["params"]) for p in model.w_out.parameters())
    for stage in "ABC":
        set_stage_trainability(model, stage)
        assert all(p.requires_grad == (stage != "A") for p in model.w_out.parameters())
    state = model.state_dict()
    del state["w_out.weight"]
    with pytest.raises(RuntimeError, match="w_out.weight"):
        model.load_state_dict(state, strict=True)


def test_residual_full_abc_resume_and_portable_ema(tmp_path):
    check_resume(tmp_path, "C_teacher_residual")
