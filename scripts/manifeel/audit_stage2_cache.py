"""Read-only real-data audit of Zarr/streamed-cache windows and complete label joins."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from adavip.manifeel.stage2_data import make_stage2_dataset, validate_complete_ram_cache
from adavip.manifeel.stage2_ablation_workspace import Stage2EpochSampler, initialize_ablation_worker
from adavip.manifeel.streaming_cache import StreamingCache
from adavip.manifeel.full_sensor_workspace import normalizer_fingerprint
from diffusion_policy.model.common.normalizer import LinearNormalizer


LABELS = '/data/wangzihao/outputs/manifeel/l1_epoch50_dropout_comparison_k64_20261102_201244/original'
CACHE = '/data/wangzihao/cache/manifeel/full_sensor_float32_9task_v1_seed42_20261017'
CONFIG = '/data/wangzihao/outputs/manifeel/full_sensor_cache_build_20261017_134757/dataset_config.yaml'


def memory(pid: int) -> dict:
    """Record proportional and anonymous memory rather than summing shared RSS."""
    names = {'Rss','Pss','Pss_Anon','Pss_File','Private_Dirty','Shared_Clean'}
    result = {'pid':pid}
    for line in Path(f'/proc/{pid}/smaps_rollup').read_text().splitlines():
        parts=line.split()
        if parts[0].rstrip(':') in names:
            result[parts[0].rstrip(':')+'_kib']=int(parts[1])
    return result


def assert_sample(a: dict, b: dict) -> None:
    """Require every observation/action/label/index/mask tensor to match exactly."""
    assert a.keys()==b.keys()
    for name in a:
        if isinstance(a[name],dict):
            assert_sample(a[name],b[name])
        else:
            assert torch.equal(a[name],b[name]), name


def main() -> None:
    """Check episode edges, random windows, worker sharing and complete cache hashes."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--production',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(exist_ok=True,parents=True)
    torch.set_num_threads(4)
    original=make_stage2_dataset(LABELS,'/data/wangzihao/datasets/manifeel',
        '/data/wangzihao/outputs/manifeel/task_embeddings/clip_vit_b32_9task.npz')
    manifest=validate_complete_ram_cache(original.source,CACHE,CONFIG,require_full_ram=False)
    assert len(original)==65265 and len(original.task_specs)==9
    assert sum(x['nbytes'] for x in manifest['arrays'])==299494742868
    index=original.store.index
    selected=set()
    episodes=0
    for task in range(9):
        rows=np.flatnonzero(index['task_index']==task)
        for episode in np.unique(index['episode_index'][rows]):
            candidates=rows[index['episode_index'][rows]==episode]
            selected.update((int(candidates[0]),int(candidates[-1])))
            episodes+=1
        selected.update(np.random.default_rng(99042+task).choice(rows,6,replace=False).tolist())
    selected=sorted(selected)
    mapped=copy.copy(original);mapped.source=copy.copy(original.source)
    cache=StreamingCache(mapped.source,CACHE,manifest,args.output)
    with np.load(Path(LABELS)/'metadata/teacher_normalizer.npz',allow_pickle=False) as archive:
        normalizer=LinearNormalizer()
        normalizer.load_state_dict({k:torch.from_numpy(archive[k].copy()) for k in archive.files},strict=True)
    contract=json.loads((args.production/'inputs_verified.json').read_text())
    assert normalizer_fingerprint(normalizer)==contract['normalizer']
    assert original.source._normalizer is None
    try:
        cache()
        assert all(isinstance(a.values,np.memmap) and not a.values.flags.writeable
                   for group in mapped.source.groups for a in group['data'].values())
        assert all(np.array_equal(a.row_map,np.arange(len(a.row_map)))
                   for group in mapped.source.groups for a in group['data'].values())
        orders={}
        for epoch in (1,2,7,200,400,600):
            a,b=Stage2EpochSampler(original,32),Stage2EpochSampler(mapped,32)
            a.epoch=b.epoch=epoch
            expected=list(a);actual=list(b);assert expected==actual and len(expected)==65265
            for cursor in (1,50,2039,2040):
                b.cursor=cursor;assert list(b)==expected[cursor*32:]
            orders[epoch]=hashlib.sha256(np.asarray(expected,dtype=np.int64).tobytes()).hexdigest()
        loader=DataLoader(Subset(mapped,selected),batch_size=None,num_workers=2,
            worker_init_fn=initialize_ablation_worker,persistent_workers=True,
            generator=torch.Generator().manual_seed(700042))
        iterator=iter(loader)
        counts=[0]*9;started=time.monotonic()
        for number,(row,actual) in enumerate(zip(selected,iterator),1):
            expected=original[row]
            assert_sample(expected,actual)
            assert_sample(normalizer.normalize(expected['obs']),normalizer.normalize(actual['obs']))
            assert torch.equal(normalizer['action'].normalize(expected['action']),
                               normalizer['action'].normalize(actual['action']))
            counts[int(index['task_index'][row])]+=1
            cache.check()
            if number%100==0:
                print(json.dumps({'event':'windows_exact','windows':number,'seconds':time.monotonic()-started}),flush=True)
        assert sum(counts)==len(selected)
        memory_report=[memory(os.getpid())]+[memory(p.pid) for p in iterator._workers]
        sample_report=dict(status='WINDOWS_PASS',windows=len(selected),episodes=episodes,per_task_windows=counts,
            all_65265_label_index_rows_validated=True,raw_vs_stream_all_fields_bitwise=True,
            normalized_obs_actions_bitwise=True,normalizer=contract['normalizer'],
            independent_epoch_order_hashes=orders,tail_batch_size=17,
            cache_bytes=manifest['nbytes'],cache_arrays=len(manifest['arrays']),
            readonly_mmap=True,identity_row_maps=True,workers=2,memory=memory_report)
        (args.output/'WINDOWS_PASS.json').write_text(json.dumps(sample_report,indent=2)+'\n')
        while not cache.finished:
            cache.check()
            if not cache.finished: time.sleep(5)
        sample_report.update(status='PASS',full_cache_hash_verification=True)
        (args.output/'PASS.json').write_text(json.dumps(sample_report,indent=2)+'\n')
        print(json.dumps(sample_report),flush=True)
    finally:
        cache.close()


if __name__=='__main__':
    main()
