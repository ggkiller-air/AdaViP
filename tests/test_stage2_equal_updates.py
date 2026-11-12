"""Scientific-clock boundaries, unequal-tail weighting and exact sampler continuation."""

import numpy as np
import pytest
import torch

from adavip.manifeel.stage2_equal_updates import (
    EqualUpdateSampler, shard_bounds, stage_for_update, weighted_ddp_loss,
)


@pytest.mark.parametrize('completed,expected', [(0,'A'),(2039,'A'),(2040,'B'),
                                              (12239,'B'),(12240,'C'),(408000,'C')])
def test_phase_uses_completed_updates(completed, expected):
    assert stage_for_update(completed) == expected


@pytest.mark.parametrize('size,world', [(49,2),(113,4),(241,4),(128,4)])
def test_unequal_tail_global_gradient(size, world):
    x = torch.arange(size, dtype=torch.float64) / size
    for stage in ('A','B','C'):
        parameter = torch.tensor(.3, dtype=torch.float64, requires_grad=True)
        a = (parameter*x-1).square()
        u = (parameter+x).square()
        reference = u.mean() if stage=='A' else a.mean()+.05*u.mean()
        expected, = torch.autograd.grad(reference, parameter)
        gradients=[]; covered=[]
        for rank in range(world):
            first,last=shard_bounds(size,rank,world);covered.extend(range(first,last))
            p=parameter.detach().clone().requires_grad_()
            local=weighted_ddp_loss((p*x[first:last]-1).square(),
                (p+x[first:last]).square(),stage,size,world)
            gradients.append(torch.autograd.grad(local,p)[0])
        assert covered==list(range(size))
        torch.testing.assert_close(torch.stack(gradients).mean(),expected,rtol=1e-14,atol=1e-14)


class ToyDataset:
    def __len__(self):return 65265
    def make_task_balanced_sampler(self,generator):
        return torch.utils.data.WeightedRandomSampler(torch.arange(1,18,dtype=torch.float64),
            len(self),replacement=True,generator=generator)


@pytest.mark.parametrize('batch,world', [(64,2),(128,4),(256,4)])
def test_sampler_exact_resume_and_full_global_draws(batch,world):
    dataset=ToyDataset()
    samplers=[EqualUpdateSampler(dataset,batch,rank,world) for rank in range(world)]
    for sampler in samplers:sampler.epoch=17
    expected=samplers[0].plan()
    ranks=[list(s) for s in samplers]
    reconstructed=[value for group in zip(*ranks) for rank in group for value in rank]
    assert reconstructed==expected and len(reconstructed)==65265
    for sampler,rank in zip(samplers,ranks):
        sampler.cursor=7
        assert list(sampler)==rank[7:]
    assert sum(len(rank[-1]) for rank in ranks)==65265%batch
