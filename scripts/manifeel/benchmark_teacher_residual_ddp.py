"""Same-node persistent DDP compute screen at unchanged global batch32."""

from __future__ import annotations

import argparse
import copy
import json
import os
import time
from dataclasses import fields
from pathlib import Path

import dill
import hydra
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from omegaconf import OmegaConf

from adavip.manifeel.decoded_sensor_cache import SourceIndexedArray
from adavip.manifeel.stage2_data import make_stage2_dataset, validate_complete_ram_cache
from adavip.manifeel.stage2_parallel import restore_optimizer_snapshot, Stage2LossModule
from adavip.manifeel.stage2_main_ablation import set_protocol_step
from adavip.manifeel.stage2_ablation_workspace import tensor_fingerprint
from scripts.manifeel.train_stage2_t2 import (make_optimizer_scheduler, set_stage_trainability,
                                            restore_policy_state, update_ema)


def main() -> None:
    """Compare baseline versus DDP without modifying any production checkpoint."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(exist_ok=True,parents=True)
    rank=int(os.environ['LOCAL_RANK']);world=int(os.environ['WORLD_SIZE'])
    assert world in (2,4) and int(os.environ['LOCAL_WORLD_SIZE'])==world
    torch.cuda.set_device(rank);device=torch.device('cuda',rank)
    torch.set_num_threads(4)
    assert 'H200' in torch.cuda.get_device_name(rank)
    dist.init_process_group('nccl')
    saved=torch.load(args.checkpoint,map_location='cpu',pickle_module=dill)
    assert saved['contract']['ablation']=='C_teacher_residual'
    policy=hydra.utils.instantiate(OmegaConf.create(saved['cfg']).policy).to(device)
    restore_policy_state(policy,saved['model'],device)
    set_stage_trainability(policy,'C');policy.train()
    dataset=make_stage2_dataset(
        '/data/wangzihao/outputs/manifeel/l1_epoch50_dropout_comparison_k64_20261102_201244/original',
        '/data/wangzihao/datasets/manifeel',
        '/data/wangzihao/outputs/manifeel/task_embeddings/clip_vit_b32_9task.npz')
    cache=Path('/data/wangzihao/cache/manifeel/full_sensor_float32_9task_v1_seed42_20261017')
    manifest=validate_complete_ram_cache(dataset.source,str(cache),
        '/data/wangzihao/outputs/manifeel/full_sensor_cache_build_20261017_134757/dataset_config.yaml',False)
    groups=[{'data':{}} for _ in dataset.task_specs]
    for item in manifest['arrays']:
        values=np.load(cache/item['filename'],mmap_mode='r',allow_pickle=False)
        groups[int(item['task_index'])]['data'][item['key']]=SourceIndexedArray(values,np.arange(len(values)),len(values))
    dataset.source.groups=groups
    indices=[int(np.flatnonzero(dataset.task_indices==i%9)[i//9]) for i in range(32)]
    batch=torch.utils.data.default_collate([dataset[i] for i in indices])
    batch={k:({n:t.to(device) for n,t in v.items()} if k=='obs' else v.to(device)) for k,v in batch.items()}
    part=slice(rank*(32//world),(rank+1)*(32//world))

    def sliced(interval):
        return {k:({n:t[interval] for n,t in v.items()} if k=='obs' else v[interval]) for k,v in batch.items()}

    def reset_rng():
        torch.set_rng_state(saved['rng']['torch'])
        torch.cuda.set_rng_state(saved['rng']['cuda'][0],rank)

    local=Stage2LossModule(policy)
    ddp=DistributedDataParallel(local,device_ids=[rank],output_device=rank,
        broadcast_buffers=False,find_unused_parameters=True,gradient_as_bucket_view=True,bucket_cap_mb=100)
    reset_rng();sample=policy.prepare_denoising_sample(batch)
    if rank==0:
        policy.zero_grad(set_to_none=True)
        a_parts,u_parts=[],[]
        for first in range(0,32,32//world):
            interval=slice(first,first+32//world)
            tensors=tuple(getattr(sample,f.name)[interval] for f in fields(sample))
            a,u=local(sliced(interval),'C',tensors);a_parts.append(a);u_parts.append(u)
        serial=torch.cat(a_parts).mean()+.05*torch.cat(u_parts).mean()
        serial.backward()
        expected={n:p.grad.detach().cpu().clone() for n,p in policy.named_parameters() if p.grad is not None}
        del a_parts,u_parts,serial
    dist.barrier()
    policy.zero_grad(set_to_none=True)
    tensors=tuple(getattr(sample,f.name)[part] for f in fields(sample))
    a,u=ddp(sliced(part),'C',tensors)
    (a.mean()+.05*u.mean()).backward()
    correct=True
    if rank==0:
        delta_sq=ref_sq=maximum=0.0
        for name,p in policy.named_parameters():
            assert (p.grad is not None)==(name in expected),name
            if p.grad is None:continue
            x,y=expected[name],p.grad.detach().cpu()
            assert torch.isfinite(y).all(),name
            difference=(x-y).double()
            delta_sq+=float(difference.square().sum());ref_sq+=float(x.double().square().sum())
            maximum=max(maximum,float(difference.abs().max()) if difference.numel() else 0.0)
        correctness=dict(matched_shard_gradient_relative_l2=(delta_sq/max(ref_sq,1e-30))**.5,
                         matched_shard_gradient_max_abs=maximum,global_batch=32,devices=world,
                         same_global_noise_and_timesteps=True)
        (args.output/'correctness.json').write_text(json.dumps(correctness,indent=2)+'\n')
        correct=correctness['matched_shard_gradient_relative_l2']<1e-4
        del expected
    flag=torch.tensor(int(correct),device=device);dist.broadcast(flag,0)
    if not bool(flag):raise RuntimeError('Matched-shard DDP gradients failed; see correctness.json')
    del sample,a,u,tensors
    policy.zero_grad(set_to_none=True)
    rates=[]
    for mode in ('single','ddp','ddp','single'):
        dist.barrier()
        restore_policy_state(policy,saved['model'],device)
        set_stage_trainability(policy,'C');policy.train();reset_rng()
        if mode=='ddp' or rank==0:
            optimizer,scheduler=make_optimizer_scheduler(policy,2040,600)
            restore_optimizer_snapshot(optimizer,saved['optimizer']);scheduler.load_state_dict(saved['scheduler'])
            holder=type('Clock',(),{})();holder.optimizer=optimizer;holder.scheduler=scheduler
            set_protocol_step(holder,max(12740,saved['global_step']))
            ema=copy.deepcopy(policy).eval().requires_grad_(False)
            for step in range(60):
                if step==10:
                    torch.cuda.synchronize();started=time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                if mode=='single':
                    result=policy.training_losses(batch,'C')
                    loss=result['action_loss']+.05*result['utility_loss']
                else:
                    sample=policy.prepare_denoising_sample(batch)
                    tensors=tuple(getattr(sample,f.name)[part] for f in fields(sample))
                    a,u=ddp(sliced(part),'C',tensors)
                    loss=a.mean()+.05*u.mean()
                assert torch.isfinite(loss)
                loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in policy.parameters() if p.requires_grad],1.0,error_if_nonfinite=True)
                optimizer.step();scheduler.step();update_ema(ema,policy,.9999)
            torch.cuda.synchronize();elapsed=time.perf_counter()-started
            del optimizer,scheduler,ema
        else:elapsed=0.0
        elapsed_tensor=torch.tensor(elapsed,device=device);dist.all_reduce(elapsed_tensor,op=dist.ReduceOp.MAX)
        if rank==0:
            row=dict(mode=mode,updates=50,seconds=float(elapsed_tensor),samples_per_second=1600/float(elapsed_tensor))
            rates.append(row);print(json.dumps(row),flush=True)
        if mode=='ddp':
            signature=tensor_fingerprint(policy.state_dict());signatures=[None]*world
            dist.all_gather_object(signatures,signature)
            assert len(set(signatures))==1,'DDP model replicas diverged'
    if rank==0:
        single=sum(r['seconds'] for r in rates if r['mode']=='single')
        parallel=sum(r['seconds'] for r in rates if r['mode']=='ddp')
        report=dict(status='COMPUTE_SCREEN_COMPLETE',implementation='DDP',devices=world,
            global_batch=32,per_device_batch=32//world,rates=rates,compute_speedup=single/parallel,
            worth_full_validation=single/parallel>=1.15,correctness=correctness,
            updated_replicas_bitwise_equal=True,production_untouched=True,
            includes_data_loading=False,requires_full_epoch_and_resume_gate_before_migration=True)
        (args.output/'REPORT.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report),flush=True)
    dist.destroy_process_group()


if __name__=='__main__':
    main()
