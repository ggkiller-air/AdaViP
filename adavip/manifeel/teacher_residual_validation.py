"""Exact Teacher action parity and zero-start residual gradient routing gates."""

from __future__ import annotations

import copy

import torch

from scripts.manifeel.train_stage2_t2 import rng_state, restore_rng, set_stage_trainability


@torch.no_grad()
def validate_teacher_parity(teacher, policy, batch: dict, device: str) -> dict:
    """Compare contexts, noise prediction and every full-sampling output bitwise."""
    saved_rng = rng_state()
    schedulers = (copy.deepcopy(teacher.noise_scheduler), copy.deepcopy(policy.noise_scheduler))
    try:
        teacher.to(device).eval()
        policy.to(device).eval()
        batch = {k: ({n: t.to(device) for n, t in v.items()} if k == "obs" else v.to(device))
                 for k, v in batch.items()}
        for role in ("obs_encoder", "normalizer", "model"):
            a, b = getattr(teacher, role).state_dict(), getattr(policy, role).state_dict()
            assert a.keys() == b.keys()
            for name in a:
                assert torch.equal(a[name], b[name]), (role, name)
        expected = teacher.obs_encoder(teacher.normalizer.normalize(batch["obs"]))
        encoded = policy.encode_stage2(batch["obs"])
        assert torch.equal(expected, encoded["z_t"]), "Teacher context differs"
        assert not bool(encoded["residual"].count_nonzero()), "Nonzero initial residual"
        assert torch.equal(expected, encoded["conditioning"]), "Residual interface differs"

        def sampling_pair():
            generator = torch.Generator(device=device).manual_seed(20261112)
            state = generator.get_state()
            before = teacher.predict_action(batch["obs"], generator=generator)
            final_state = generator.get_state()
            generator.set_state(state)
            after = policy.predict_action(batch["obs"], generator=generator)
            assert torch.equal(final_state, generator.get_state())
            for key in before:
                assert torch.equal(before[key], after[key]), key
                assert bool(torch.isfinite(after[key]).all())

        # First compare the fresh, portable evaluation state (both schedules on CPU).
        sampling_pair()
        sample = teacher.prepare_denoising_sample(batch)
        devices_after_teacher_only = [str(m.noise_scheduler.alphas_cumprod.device)
                                      for m in (teacher, policy)]
        # diffusers.add_noise mutates alphas_cumprod's device. Match numerical state
        # by preparing the identical sample on both sides, not by relaxing parity.
        paired = policy.prepare_denoising_sample(batch, noise=sample.noise, timesteps=sample.timesteps)
        assert torch.equal(sample.noisy_action, paired.noisy_action)
        a = teacher.model(sample.noisy_action, sample.timesteps, global_cond=expected)
        b = policy.model(sample.noisy_action, sample.timesteps, global_cond=encoded["conditioning"])
        assert torch.equal(a, b), "Denoising prediction differs"
        sampling_pair()  # Also compare both warmed schedules, as after training.
        return dict(status="PASS", teacher_modules_exact=True, context_exact=True,
                    zero_residual=True, denoising_prediction_exact=True,
                    full_predict_action_exact=True, sampling_rng_exact=True,
                    fresh_scheduler_sampling_exact=True, warmed_scheduler_sampling_exact=True,
                    scheduler_devices_after_teacher_only_add_noise=devices_after_teacher_only,
                    batch_size=len(expected), inference_steps=policy.num_inference_steps,
                    deterministic_algorithms=torch.are_deterministic_algorithms_enabled())
    finally:
        teacher.noise_scheduler, policy.noise_scheduler = schedulers
        restore_rng(saved_rng)


def validate_residual_routes(policy, batch: dict) -> dict:
    """Temporarily update only W_out, then restore weights/RNG after all stage checks."""
    from adavip.manifeel.stage2_ablation_workspace import parameter_report

    saved_rng = rng_state()
    saved_w = copy.deepcopy(policy.w_out.state_dict())
    trainable = {n: p.requires_grad for n, p in policy.named_parameters()}
    mode = policy.training
    report = {}
    initially_zero = not any(bool(p.count_nonzero()) for p in policy.w_out.parameters())

    def active(module):
        return any(p.grad is not None and bool(p.grad.count_nonzero()) for p in module.parameters())

    try:
        for stage in ("A", "B", "C"):
            policy.w_out.load_state_dict(saved_w, strict=True)
            set_stage_trainability(policy, stage)
            policy.train()
            policy.zero_grad(set_to_none=True)
            losses = policy.training_losses(batch, stage)
            assert bool(torch.isfinite(losses["utility_loss"] + losses["action_loss"]))
            losses["utility_loss"].backward(retain_graph=stage != "A")
            assert active(policy.utility_predictor)
            assert all(p.grad is None for n, p in policy.named_parameters()
                       if not n.startswith("utility_predictor.")), "Utility detach failed"
            if stage == "A":
                assert all(not p.requires_grad for n, p in policy.named_parameters()
                           if not n.startswith("utility_predictor.")), "Actor trainable in A"
                report[stage] = parameter_report(policy, gradients=True)
                continue
            policy.zero_grad(set_to_none=True)
            losses["action_loss"].backward()
            assert all(p.grad is None for p in policy.utility_predictor.parameters())
            assert all(p.grad is None for p in policy.parameters() if not p.requires_grad)
            assert active(policy.model) and active(policy.w_out)
            upstream = (policy.cross_attention, policy.query_projection,
                        policy.fusion_ff, policy.language_token_projection)
            if initially_zero:
                assert all(not active(m) for m in upstream), "Zero W_out leaked gradient"
            for module in policy.obs_encoder.image_encoders.values():
                assert active(module) == (stage == "C")
            report[stage] = {"initial": parameter_report(policy, gradients=True),
                             "zero_w_out": initially_zero}
            # Exercise a genuine optimizer update; never alter formal initialization.
            optimizer = torch.optim.SGD(policy.w_out.parameters(), lr=1e-3)
            optimizer.step()
            policy.zero_grad(set_to_none=True)
            after = policy.training_losses(batch, stage)
            after["action_loss"].backward()
            assert all(active(m) for m in upstream), "Residual branch remains blocked after W_out update"
            assert all(p.grad is None for p in policy.utility_predictor.parameters())
            report[stage]["after_w_out_update"] = parameter_report(policy, gradients=True)
    finally:
        policy.w_out.load_state_dict(saved_w, strict=True)
        policy.zero_grad(set_to_none=True)
        for name, p in policy.named_parameters():
            p.requires_grad_(trainable[name])
        policy.train(mode)
        restore_rng(saved_rng)
    return report
