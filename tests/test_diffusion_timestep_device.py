"""Regression: diffusers yields CPU scalar timesteps for CUDA denoising."""
import importlib.util
from pathlib import Path
import pytest
import torch

spec = importlib.util.spec_from_file_location("tested_unet", Path(__file__).resolve().parents[1] / "adavip/tabero/unet.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("timestep", [5, torch.tensor(5), torch.tensor([5, 6])])
def test_scheduler_timestep_device(device, timestep):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("Requires an allocated GPU")
    model = module.ConditionalUnet1D(input_dim=8, global_cond_dim=16,
        diffusion_step_embed_dim=16, down_dims=(16, 32, 64)).to(device).eval()
    sample = torch.randn(2, 16, 8, device=device)
    condition = torch.randn(2, 16, device=device)
    with torch.no_grad():
        output = model(sample, timestep, condition)
    assert output.shape == sample.shape
    assert output.device == sample.device
    assert torch.isfinite(output).all()
