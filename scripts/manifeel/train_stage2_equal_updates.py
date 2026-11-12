"""Fresh Teacher-residual batch comparisons on the original 1,224,000-update clock."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import signal
from pathlib import Path

import cv2
import dill
import hydra
import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf

from adavip.manifeel.decoded_sensor_cache import SourceIndexedArray
from adavip.manifeel.full_sensor_workspace import normalizer_fingerprint
from adavip.manifeel.stage2_ablation_workspace import tensor_fingerprint
from adavip.manifeel.stage2_data import make_stage2_dataset, validate_complete_ram_cache
from adavip.manifeel.stage2_equal_updates import EqualUpdateTrainer, TOTAL_UPDATES, REFERENCE_STEPS, FIXED_EPOCHS
from adavip.manifeel.stage2_main_ablation import FULL_INITIALIZATION_SHA256, audit_pruned_initialization, set_protocol_step
from adavip.manifeel.streaming_cache import StreamingCache
from scripts.manifeel.train_stage2_l1_s0 import load_teacher, TEACHER_SHA256, LABEL_SHA256

TEACHER = '/data/wangzihao/outputs/manifeel/language_conditioning_t2_batch64_ep500_seed42_20261028_204459/l1/checkpoints/epoch50.ckpt'
LABELS = '/data/wangzihao/outputs/manifeel/l1_epoch50_dropout_comparison_k64_20261102_201244/original'
CACHE = '/data/wangzihao/cache/manifeel/full_sensor_float32_9task_v1_seed42_20261017'
CACHE_CONFIG = '/data/wangzihao/outputs/manifeel/full_sensor_cache_build_20261017_134757/dataset_config.yaml'


def assert_tree_equal(a, b, path='root'):
    """Compare full optimizer/model/RNG structures, with no numerical tolerance."""
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b), path
    elif isinstance(a, np.ndarray):
        assert np.array_equal(a, b), path
    elif isinstance(a, dict):
        assert a.keys() == b.keys(), path
        for key in a:
            assert_tree_equal(a[key], b[key], path + '.' + str(key))
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            assert_tree_equal(x, y, path + '.' + str(i))
    else:
        assert a == b, path


def smoke(trainer, phase):
    """Check real phase updates, a full data pass, and a fresh-process exact resume."""
    out = trainer.output
    if phase == 'first':
        roles = ('obs_encoder', 'model', 'w_out', 'query_projection', 'utility_predictor')
        def signatures():
            return {name:tensor_fingerprint(getattr(trainer.policy, name).state_dict()) for name in roles}
        before = signatures()
        trainer.run(stop_after=2)
        after_a = signatures()
        assert all(before[n] == after_a[n] for n in roles if n != 'utility_predictor')
        assert before['utility_predictor'] != after_a['utility_predictor']
        set_protocol_step(trainer, 2040)
        trainer.data_epoch, trainer.cursor = divmod(2040, trainer.steps_per_data_epoch)
        trainer.data_epoch += 1
        trainer.run(stop_after=2)
        after_b = signatures()
        assert after_b['obs_encoder'] == after_a['obs_encoder']
        assert after_b['model'] != after_a['model'] and after_b['w_out'] != after_a['w_out']
        set_protocol_step(trainer, 12240)
        trainer.data_epoch, trainer.cursor = divmod(12240, trainer.steps_per_data_epoch)
        trainer.data_epoch += 1
        trainer.run(stop_after=trainer.steps_per_data_epoch + 10)
        after_c = signatures()
        assert all(after_c[n] != after_b[n] for n in roles)
        trainer.assert_replicas()
        if trainer.rank == 0:
            os.link(out/'checkpoints/latest.ckpt', out/'resume_seed.ckpt')
        dist.barrier()
        trainer.run(stop_after=2)
        if trainer.rank == 0:
            os.link(out/'checkpoints/latest.ckpt', out/'expected_next.ckpt')
            (out/'PHASES_PASS.json').write_text(json.dumps(dict(status='PASS',
                stage_a_utility_only=True, stage_b_encoder_frozen=True,
                stage_c_all_roles_updated=True, full_data_epoch=True,
                tail_size=len(trainer.dataset) % trainer.batch_size,
                batch=trainer.batch_size, world=trainer.world), indent=2)+'\n')
    else:
        trainer.restore(out/'resume_seed.ckpt')
        trainer.run(stop_after=2)
        if trainer.rank == 0:
            expected = torch.load(out/'expected_next.ckpt', map_location='cpu', pickle_module=dill)
            actual = torch.load(out/'checkpoints/latest.ckpt', map_location='cpu', pickle_module=dill)
            assert_tree_equal(expected, actual)
            del expected, actual
            from adavip.manifeel.full_sensor_evaluation import load_evaluation_policy
            rows = [int(np.flatnonzero(trainer.dataset.task_indices == task)[0]) for task in range(9)]
            batch = torch.utils.data.default_collate([trainer.dataset[i] for i in rows])
            observations = {key:value.to(trainer.device) for key,value in batch['obs'].items()}
            with torch.no_grad():
                for weights, reference in [('raw',trainer.policy),('ema',trainer.ema)]:
                    restored, _, _ = load_evaluation_policy(out/'checkpoints/latest.ckpt',
                                                            device=str(trainer.device),weights=weights)
                    assert tensor_fingerprint(reference.state_dict()) == tensor_fingerprint(restored.state_dict())
                    reference.eval()
                    # Compare both with the portable loader's fresh diffusion scheduler.
                    scheduler = reference.noise_scheduler
                    reference.noise_scheduler = copy.deepcopy(restored.noise_scheduler)
                    before = reference.predict_action(observations,
                        generator=torch.Generator(device=trainer.device).manual_seed(20261112))
                    after = restored.predict_action(observations,
                        generator=torch.Generator(device=trainer.device).manual_seed(20261112))
                    assert_tree_equal(before, after)
                    reference.noise_scheduler = scheduler
                    del restored
            report = json.loads((out/'PHASES_PASS.json').read_text())
            report.update(fresh_process_resume_bitwise=True, raw_ema_adam_scheduler_rng_exact=True,
                          sampler_cursor_task_statistics_exact=True,
                          portable_raw_ema_full_sampling_bitwise=True,
                          total_updates=TOTAL_UPDATES, comparison_updates=[408000,816000])
            (out/'SMOKE_PASS.json').write_text(json.dumps(report, indent=2)+'\n')
    dist.barrier()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch', type=int, choices=(64,128,256), required=True)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--smoke-phase', choices=('first','restore'))
    args = parser.parse_args()
    rank, world = int(os.environ['LOCAL_RANK']), int(os.environ['WORLD_SIZE'])
    assert world == (2 if args.batch == 64 else 4)
    allocated = sorted(os.sched_getaffinity(0))
    rank_cores = allocated[rank::world]
    assert len(rank_cores) >= args.workers + 4
    worker_cores = rank_cores[-args.workers:] if args.workers else []
    main_cores = rank_cores[:-args.workers] if args.workers else rank_cores
    os.environ['EQUAL_WORKER_CORES'] = ','.join(str(core) for core in worker_cores)
    os.sched_setaffinity(0, set(main_cores))
    torch.cuda.set_device(rank)
    assert 'H200' in torch.cuda.get_device_name(rank)
    dist.init_process_group('nccl')
    cv2.setNumThreads(0); torch.set_num_threads(4)
    torch.manual_seed(42); random.seed(42); np.random.seed(42)
    if args.smoke_phase:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
    dataset = make_stage2_dataset(LABELS, '/data/wangzihao/datasets/manifeel',
        '/data/wangzihao/outputs/manifeel/task_embeddings/clip_vit_b32_9task.npz')
    assert len(dataset) == 65265 and len(dataset.task_specs) == 9
    assert dataset.store.manifest['sha256'] == LABEL_SHA256
    manifest = validate_complete_ram_cache(dataset.source, CACHE, CACHE_CONFIG, require_full_ram=False)
    utility = np.asarray(dataset.store.utility, dtype=np.float64)
    rms = np.maximum(np.sqrt(np.square(utility).mean(axis=0)), 1e-6)
    teacher, cfg = load_teacher(Path(TEACHER), Path(LABELS))
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    cfg.policy._target_ = 'adavip.manifeel.stage2_l1_policy.Stage2L1Policy'
    cfg.policy.encoder_strategy = 's0'; cfg.policy.utility_rms = rms.tolist()
    policy = hydra.utils.instantiate(cfg.policy)
    policy.initialize_from_teacher(teacher)
    full = policy.state_dict()
    assert tensor_fingerprint(full) == FULL_INITIALIZATION_SHA256
    rng = torch.get_rng_state().clone()
    policy.apply_ablation('C_teacher_residual')
    audit = audit_pruned_initialization(full, policy)
    assert tensor_fingerprint({k:v for k,v in policy.state_dict().items() if not k.startswith('w_out.')}) == '21db0ba749cc21041909ec85f68de14bae63f7693063b72eb03929a22b4fbf03'
    del full
    policy.initialize_from_teacher(teacher)
    assert tensor_fingerprint(policy.model.state_dict()) == tensor_fingerprint(teacher.model.state_dict())
    assert torch.equal(rng, torch.get_rng_state())
    assert not any(bool(p.count_nonzero()) for p in policy.w_out.parameters())
    cfg.policy.ablation = 'C_teacher_residual'
    audit.update(residual_initialization_sha256=tensor_fingerprint(policy.state_dict()),
                 teacher_ema_unet_sha256=tensor_fingerprint(teacher.model.state_dict()))
    args.output.mkdir(exist_ok=True, parents=True)
    (args.output/f'affinity_rank{rank}.json').write_text(json.dumps(dict(
        main_cores=main_cores, worker_cores=worker_cores, allocated_cores=allocated))+'\n')
    if rank == 0 and args.smoke_phase == 'first':
        from adavip.manifeel.teacher_residual_validation import validate_teacher_parity, validate_residual_routes
        rows = [int(np.flatnonzero(dataset.task_indices == t)[0]) for t in range(9)]
        batch = torch.utils.data.default_collate([dataset[i] for i in rows])
        report = validate_teacher_parity(teacher, policy, batch, f'cuda:{rank}')
        (args.output/'teacher_parity.json').write_text(json.dumps(report, indent=2)+'\n')
        batch = {k:({n:t.to(rank) for n,t in v.items()} if k=='obs' else v.to(rank)) for k,v in batch.items()}
        routes = validate_residual_routes(policy, batch)
        (args.output/'gradient_routes.json').write_text(json.dumps(routes, indent=2)+'\n')
    del teacher
    dist.barrier()
    # Independent, persisted rank streams avoid duplicated diffusion noise.
    torch.cuda.manual_seed(42 + rank)
    groups = [{'data':{}} for _ in dataset.task_specs]
    identities = []
    for record in manifest['arrays']:
        path = Path(CACHE)/record['filename']; values = np.load(path, mmap_mode='r', allow_pickle=False)
        groups[int(record['task_index'])]['data'][record['key']] = SourceIndexedArray(values, np.arange(len(values)), len(values))
        st = path.stat(); identities.append((record['filename'],st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns))
    dataset.source.groups = groups; dataset.source.verified_ram_cache = False
    cache_signatures = [None] * world
    dist.all_gather_object(cache_signatures, hashlib.sha256(json.dumps(identities).encode()).hexdigest())
    assert len(set(cache_signatures)) == 1
    root = Path(__file__).resolve().parents[2]
    digest = hashlib.sha256()
    for directory in ('adavip/manifeel','scripts/manifeel'):
        for path in sorted((root/directory).glob('*.py')):
            digest.update(str(path.relative_to(root)).encode()); digest.update(path.read_bytes())
    contract = dict(ablation='C_teacher_residual', experiment='equal_optimizer_updates',
        variant='stage2_transformer', encoder_variant='l1', encoder_strategy='s0',
        batch_size=args.batch, world_size=world, seed=42, rank_cuda_seed='42+rank',
        total_updates=TOTAL_UPDATES, reference_steps_per_epoch=REFERENCE_STEPS,
        stage_a_updates=2040, stage_b_updates=10200, stage_c_updates=1211760,
        comparison_updates=[408000,816000], checkpoint_reference_epochs=list(FIXED_EPOCHS),
        teacher_sha256=TEACHER_SHA256, labels_sha256=LABEL_SHA256, cache_contract=manifest['contract'],
        cache_bytes=manifest['nbytes'], cache_file_identity=cache_signatures[0],
        task_embedding_sha256=dataset.source.manifest['task_embedding_sha256'],
        task_ids=[spec.task_id for spec in dataset.task_specs],
        normalizer=normalizer_fingerprint(policy.normalizer), initialization=audit,
        encoder_lr=8e-5, utility_lr=1e-4, action_lr=1e-4, gradient_clip_norm=1.0,
        utility_weight=.05, ema_decay=.9999, warmup_updates=500, encoder_ramp_updates=500,
        source_sha256=digest.hexdigest(), smoke=bool(args.smoke_phase))
    if rank == 0:
        for name, value in [('contract.json',contract),('inputs_verified.json',contract),
                            ('initialization_audit.json',audit),('architecture.json',OmegaConf.to_container(cfg,resolve=True))]:
            (args.output/name).write_text(json.dumps(value, indent=2)+'\n')
    trainer = EqualUpdateTrainer(policy, dataset, OmegaConf.to_container(cfg,resolve=True),
                                contract, args.output, workers=args.workers)
    for sig in (signal.SIGUSR1,signal.SIGTERM,signal.SIGINT):
        signal.signal(sig, lambda *_:setattr(trainer,'stop_requested',True))
    if args.smoke_phase:
        smoke(trainer, args.smoke_phase)
    else:
        if args.resume:
            trainer.restore(args.resume)
        elif (args.output/'checkpoints/latest.ckpt').exists():
            raise FileExistsError('Explicit resume required')
        monitor = StreamingCache(dataset.source, CACHE, manifest, args.output) if rank == 0 else None
        try:
            if monitor is not None: monitor()
            trainer.run(monitor=monitor)
        finally:
            if monitor is not None: monitor.close()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
