"""Bounded two/four-H200 compute screening; never writes to the production run."""

from __future__ import annotations

import argparse
import copy
import json
import os
import time
from pathlib import Path

import dill
import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

from adavip.manifeel.decoded_sensor_cache import SourceIndexedArray
from adavip.manifeel.stage2_data import make_stage2_dataset, validate_complete_ram_cache
from adavip.manifeel.stage2_parallel import restore_optimizer_snapshot, MultiDeviceLosses, Stage2LossModule
from scripts.manifeel.train_stage2_t2 import (make_optimizer_scheduler, set_stage_trainability,
                                            update_ema, restore_policy_state, rng_state, restore_rng)


def main() -> None:
    """Compare identical single/parallel workloads and reject an unhelpful compute speedup."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--devices", type=int, choices=(2, 4), required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=True, parents=True)
    assert torch.cuda.device_count() == args.devices
    devices = list(range(args.devices))
    for i in devices:
        assert 'H200' in torch.cuda.get_device_name(i)
        assert torch.cuda.mem_get_info(i)[0] > 80 * 2**30
    torch.set_num_threads(4)
    torch.manual_seed(42)
    stable = args.output / "seed.ckpt"
    if not stable.exists():
        os.link(args.checkpoint, stable)
    saved = torch.load(stable, map_location="cpu", pickle_module=dill)
    assert saved["contract"]["ablation"] == "C_teacher_residual"
    cfg = OmegaConf.create(saved["cfg"])
    policy = hydra.utils.instantiate(cfg.policy).cuda(0)
    restore_policy_state(policy, saved["model"], torch.device("cuda:0"))
    dataset = make_stage2_dataset(
        '/data/wangzihao/outputs/manifeel/l1_epoch50_dropout_comparison_k64_20261102_201244/original',
        '/data/wangzihao/datasets/manifeel',
        '/data/wangzihao/outputs/manifeel/task_embeddings/clip_vit_b32_9task.npz')
    cache = Path('/data/wangzihao/cache/manifeel/full_sensor_float32_9task_v1_seed42_20261017')
    manifest = validate_complete_ram_cache(dataset.source, str(cache),
        '/data/wangzihao/outputs/manifeel/full_sensor_cache_build_20261017_134757/dataset_config.yaml',
        require_full_ram=False)
    groups = [{"data": {}} for _ in dataset.task_specs]
    for item in manifest['arrays']:
        values = np.load(cache / item['filename'], mmap_mode='r', allow_pickle=False)
        groups[int(item['task_index'])]['data'][item['key']] = SourceIndexedArray(values, np.arange(len(values)), len(values))
    dataset.source.groups = groups
    # Keep normal per-window value checks; this small screen does not claim full-cache validation.
    indices = [int(np.flatnonzero(dataset.task_indices == i % 9)[i // 9]) for i in range(32)]
    batch = torch.utils.data.default_collate([dataset[i] for i in indices])
    batch = {k: ({n: t.cuda(0) for n, t in v.items()} if k == 'obs' else v.cuda(0)) for k, v in batch.items()}
    set_stage_trainability(policy, 'C')
    policy.train()
    parallel = MultiDeviceLosses(policy, device_ids=devices)
    sample = policy.prepare_denoising_sample(batch)
    policy.zero_grad(set_to_none=True)
    reference = policy.training_losses(batch, 'C', sample)
    (reference['action_loss'] + .05 * reference['utility_loss']).backward()
    gradients = {n: p.grad.detach().cpu().clone() for n,p in policy.named_parameters() if p.grad is not None}
    policy.zero_grad(set_to_none=True)
    actual = parallel(batch, 'C', sample)
    (actual['action_loss'] + .05 * actual['utility_loss']).backward()
    # Per-device GEMM/conv and inter-device sums differ in rounding from batch32.
    # Record norms and absolute discrepancies; do not claim single/parallel bitwise parity.
    delta_sq = ref_sq = 0.0
    maximum = 0.0
    for name,p in policy.named_parameters():
        assert (p.grad is not None) == (name in gradients), name
        if p.grad is None: continue
        a, b = gradients[name], p.grad.detach().cpu()
        assert torch.isfinite(b).all(), name
        diff = (a-b).double()
        delta_sq += float(diff.square().sum());ref_sq += float(a.double().square().sum())
        maximum = max(maximum, float(diff.abs().max()) if diff.numel() else 0.0)
    correctness = dict(gradient_relative_l2=(delta_sq / max(ref_sq, 1e-30))**.5,
                       gradient_max_abs=maximum,
                       action_loss_single=float(reference['action_loss']),action_loss_parallel=float(actual['action_loss']),
                       utility_loss_single=float(reference['utility_loss']),utility_loss_parallel=float(actual['utility_loss']))
    (args.output/'correctness.json').write_text(json.dumps(correctness,indent=2)+'\n')
    # Localize batch-shape arithmetic separately from the device adapter. Match
    # each shard's batch shape on one GPU and compare with the parallel reduction.
    parallel_gradients = {n: p.grad.detach().cpu().clone() for n,p in policy.named_parameters() if p.grad is not None}
    del gradients
    policy.zero_grad(set_to_none=True)
    from dataclasses import fields
    local = Stage2LossModule(policy)
    actions, utilities = [], []
    offset = 0
    for chunk in torch.arange(len(batch['action'])).chunk(args.devices):
        part = slice(offset, offset + len(chunk)); offset += len(chunk)
        sub = {k: ({n:t[part] for n,t in v.items()} if k=='obs' else v[part]) for k,v in batch.items()}
        tensors = tuple(getattr(sample,f.name)[part] for f in fields(sample))
        a,u = local(sub,'C',tensors);actions.append(a);utilities.append(u)
    serial_loss = torch.cat(actions).mean() + .05*torch.cat(utilities).mean()
    serial_loss.backward()
    delta_sq = ref_sq = 0.0
    maximum = 0.0
    for name,p in policy.named_parameters():
        assert (p.grad is not None) == (name in parallel_gradients), name
        if p.grad is None: continue
        a,b = p.grad.detach().cpu(),parallel_gradients[name]
        diff = (a-b).double()
        delta_sq += float(diff.square().sum());ref_sq += float(a.double().square().sum())
        maximum=max(maximum,float(diff.abs().max()) if diff.numel() else 0.0)
    correctness['matched_shard_gradient_relative_l2']=(delta_sq/max(ref_sq,1e-30))**.5
    correctness['matched_shard_gradient_max_abs']=maximum
    correctness['full_batch_vs_split_bitwise_equivalence_claimed']=False
    (args.output/'correctness.json').write_text(json.dumps(correctness,indent=2)+'\n')
    assert correctness['matched_shard_gradient_relative_l2'] < 1e-4, correctness
    del parallel_gradients, reference, actual, sample, actions, utilities, serial_loss

    policy.zero_grad(set_to_none=True)
    rates = []
    # ABBA order reduces first-run autotune and temporal-load bias. Include Adam,
    # EMA and global clipping; measure separately from disk/loader/checkpoint work.
    for mode in ('single','parallel','parallel','single'):
        restore_policy_state(policy, saved['model'], torch.device('cuda:0'))
        set_stage_trainability(policy,'C');policy.train()
        optimizer,scheduler=make_optimizer_scheduler(policy,2040,600)
        restore_optimizer_snapshot(optimizer,saved['optimizer']);scheduler.load_state_dict(saved['scheduler'])
        # Compare fully unfrozen Stage C even if the captured checkpoint is from B.
        from adavip.manifeel.stage2_main_ablation import set_protocol_step
        holder=type('Clock',(),{})();holder.optimizer=optimizer;holder.scheduler=scheduler
        set_protocol_step(holder,max(12740,saved['global_step']))
        ema=copy.deepcopy(policy).eval().requires_grad_(False)
        restore_rng(saved['rng'])
        for index in range(60):
            if index==10:
                for d in devices: torch.cuda.synchronize(d)
                started=time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            result=policy.training_losses(batch,'C') if mode=='single' else parallel(batch,'C')
            loss=result['action_loss']+.05*result['utility_loss']
            assert torch.isfinite(loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in policy.parameters() if p.requires_grad],1.0,error_if_nonfinite=True)
            optimizer.step();scheduler.step();update_ema(ema,policy,.9999)
        for d in devices: torch.cuda.synchronize(d)
        elapsed=time.perf_counter()-started
        row=dict(mode=mode,updates=50,seconds=elapsed,samples_per_second=1600/elapsed)
        rates.append(row);print(json.dumps(row),flush=True)
        del optimizer,scheduler,ema
    single=sum(r['seconds'] for r in rates if r['mode']=='single')
    dual_time=sum(r['seconds'] for r in rates if r['mode']=='parallel')
    report=dict(status='COMPUTE_SCREEN_COMPLETE',correctness=correctness,rates=rates,
                compute_speedup=single/dual_time,worth_full_validation=single/dual_time>=1.15,
                production_untouched=True,global_batch=32,devices=args.devices, per_device_batch=32//args.devices,
                captured_step=saved['global_step'],includes_data_loading=False,
                requires_full_epoch_and_resume_gate_before_migration=True)
    (args.output/'REPORT.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    main()
