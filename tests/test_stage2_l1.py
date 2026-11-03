"""L1 inheritance, S0 gradients, exact A/B/C continuation and portable inference."""

import torch
from omegaconf import OmegaConf

from adavip.manifeel.language_conditioned_obs_encoder import LanguageConditionedObsEncoder
from adavip.manifeel.stage2_l1_policy import Stage2L1Policy
from adavip.manifeel.stage2_ablation_workspace import Stage2AblationTrainer, validate_gradient_routes
from adavip.manifeel.full_sensor_evaluation import load_evaluation_policy
from test_stage2_policy import make_policy, make_batch, TinyImageEncoder
from test_stage2_evaluation import small_config
from test_stage2_encoder_ablation import ToyDataset
from scripts.manifeel.train_stage2_t2 import set_stage_trainability


def policy():
    """Use the real L1 encoder/head with tiny image and diffusion backbones."""
    torch.manual_seed(42)
    base = make_policy()
    encoder = LanguageConditionedObsEncoder(base.obs_encoder.shape_meta, image_feature_dim=8,
                                            use_group_norm=False, rgb_model=TinyImageEncoder())
    model = Stage2L1Policy(encoder, base.noise_scheduler, horizon=16, n_obs_steps=2,
                          n_action_steps=8, num_inference_steps=8, down_dims=(16, 32),
                          diffusion_step_embed_dim=16, n_groups=8)
    model.set_normalizer(base.normalizer)
    return model


def test_l1_context_identity_and_gradient_routes():
    model = policy()
    batch = make_batch()
    encoded = model.encode_stage2(batch["obs"])
    expected = model.obs_encoder(model.normalizer.normalize(batch["obs"]))
    torch.testing.assert_close(encoded["z_t"], expected, rtol=0, atol=0)
    assert encoded["tokens"].shape == (2, 7, 256)
    assert encoded["u_hat"].shape == (2, 9)
    for i, film in enumerate(model.film_hypernets):
        token = encoded["tokens"][:, i]
        torch.testing.assert_close(film(token, torch.randn(2, 128)), token, rtol=0, atol=0)
    validate_gradient_routes(model, batch)
    for stage in ("A", "B", "C"):
        set_stage_trainability(model, stage)
        assert all(p.requires_grad == (stage == "C") for p in model.obs_encoder.parameters())
        assert all(p.requires_grad == (stage != "A") for p in model.language_token_projection.parameters())
    teacher = policy()
    model.initialize_from_teacher(teacher)
    for key, value in teacher.obs_encoder.state_dict().items():
        torch.testing.assert_close(model.obs_encoder.state_dict()[key], value, rtol=0, atol=0)


def make_trainer(output, stop=0):
    model = policy()
    cfg = small_config()
    cfg.policy.obs_encoder._target_ = "adavip.manifeel.language_conditioned_obs_encoder.LanguageConditionedObsEncoder"
    cfg.policy.obs_encoder.variant = "l1"
    return Stage2AblationTrainer(
        model, ToyDataset(), output,
        dict(variant="stage2_transformer", encoder_variant="l1", encoder_strategy="s0"),
        OmegaConf.to_container(cfg, resolve=True), lambda: 1234,
        device="cpu", batch_size=4, epochs=7, workers=0, stop_after_steps=stop)


def test_l1_exact_abc_resume_and_inference(tmp_path):
    full = make_trainer(tmp_path / "full")
    full.run()
    stopped = make_trainer(tmp_path / "resumed", stop=3)
    stopped.run()
    resumed = make_trainer(stopped.output)
    resumed.run(stopped.output / "checkpoints/latest.ckpt")
    assert resumed.global_step == full.global_step == 14
    for role in ("policy", "ema"):
        for key, value in getattr(full, role).state_dict().items():
            torch.testing.assert_close(value, getattr(resumed, role).state_dict()[key], rtol=0, atol=0)
    for i, values in full.optimizer.state_dict()["state"].items():
        for key, value in values.items():
            torch.testing.assert_close(value, resumed.optimizer.state_dict()["state"][i][key], rtol=0, atol=0)
    assert full.scheduler.state_dict() == resumed.scheduler.state_dict()
    assert full.cumulative_counts == resumed.cumulative_counts
    restored, _, _ = load_evaluation_policy(resumed.output / "checkpoints/latest.ckpt")
    assert isinstance(restored, Stage2L1Policy)
    obs = make_batch()["obs"]
    with torch.no_grad():
        before = resumed.ema.predict_action(obs, generator=torch.Generator().manual_seed(13))
        after = restored.predict_action(obs, generator=torch.Generator().manual_seed(13))
    torch.testing.assert_close(before["action"], after["action"], rtol=0, atol=0)
