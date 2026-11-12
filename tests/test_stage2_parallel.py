"""Global loss reduction and noise draw accounting for the two-device adapter."""

import pytest
import torch

from adavip.manifeel.stage2_parallel import MultiDeviceLosses
from scripts.manifeel.train_stage2_t2 import set_stage_trainability
from test_stage2_l1 import policy
from test_stage2_policy import make_batch


@pytest.mark.parametrize('stage', ['A', 'B', 'C'])
@pytest.mark.parametrize('count', [1, 2, 5])
def test_loss_adapter_keeps_global_reduction_and_rng(stage, count):
    model = policy()
    model.apply_ablation('C_teacher_residual')
    set_stage_trainability(model,stage)
    batch = make_batch(count)
    saved = torch.get_rng_state().clone()
    expected = model.training_losses(batch,stage)
    expected_rng = torch.get_rng_state().clone()
    torch.set_rng_state(saved)
    actual = MultiDeviceLosses(model,device_ids=[])(batch,stage)
    assert torch.equal(expected_rng,torch.get_rng_state())
    for key in ('action_loss','utility_loss','action_loss_per_sample','utility_loss_per_sample'):
        torch.testing.assert_close(actual[key],expected[key],rtol=0,atol=0)


def test_benchmark_optimizer_reset_does_not_mutate_cpu_step_snapshot():
    import copy
    from adavip.manifeel.stage2_parallel import restore_optimizer_snapshot
    p=torch.nn.Parameter(torch.ones(2))
    optimizer=torch.optim.AdamW([p],lr=.01)
    p.sum().backward();optimizer.step();optimizer.zero_grad()
    saved=copy.deepcopy(optimizer.state_dict())
    before=copy.deepcopy(saved)
    restore_optimizer_snapshot(optimizer,saved)
    p.sum().backward();optimizer.step()
    for key,value in before['state'][0].items():
        assert torch.equal(value,saved['state'][0][key]),key
    assert optimizer.state[p]['step']==2
    assert saved['state'][0]['step']==1
