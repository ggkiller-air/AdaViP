"""Exact decoded samples, asynchronous corruption detection and transport resume."""
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader
from adavip.manifeel.decoded_sensor_cache import build_decoded_cache
from adavip.manifeel.stage2_data import Stage2SourceDataset
from adavip.manifeel.streaming_cache import StreamingCache, preserve_training_contract
from test_full_sensor_stage0 import dataset_arguments
from test_stage2_main_ablation import trainer


def test_streaming_exact_samples_and_corruption(dataset_arguments, tmp_path):
    source = Stage2SourceDataset(**dataset_arguments)
    expected = [source[i] for i in range(len(source))]
    cache = tmp_path / 'cache'
    manifest = build_decoded_cache(source, str(cache), workers=2)
    controller = StreamingCache(source, str(cache), manifest, tmp_path / 'run')
    try:
        assert controller() is controller
        assert isinstance(source.groups[0]['data']['front'].values, np.memmap)
        assert not source.groups[0]['data']['front'].values.flags.writeable
        actual = list(DataLoader(source, batch_size=None, num_workers=2))
        for a, b in zip(expected, actual):
            for key in a['obs']:
                torch.testing.assert_close(a['obs'][key], b['obs'][key], rtol=0, atol=0)
            torch.testing.assert_close(a['action'], b['action'], rtol=0, atol=0)
        assert len(actual) == len(expected)
        controller.process.wait(timeout=30)
        controller.check()
        assert controller.finished
    finally:
        controller.close()
    values = np.load(cache / 'task0_front.npy', mmap_mode='r+')
    values.flat[0] += .01
    values.flush()
    broken = StreamingCache(source, str(cache), manifest, tmp_path / 'broken')
    try:
        broken()
        broken.process.wait(timeout=30)
        with pytest.raises(RuntimeError, match='verification failed'):
            broken.check()
    finally:
        broken.close()


def test_streaming_monitor_preserves_exact_resume(tmp_path):
    reference = trainer(tmp_path / 'reference', 'C')
    reference.run()
    stopped = trainer(tmp_path / 'resumed', 'C', stop=3)
    stopped.run()
    resumed = trainer(stopped.output, 'C')
    resumed.workers = 2
    monitor = SimpleNamespace(manifest={'nbytes': 1}, check=lambda: None)
    resumed.cache_loader = lambda: monitor
    resumed.run(stopped.output / 'checkpoints/latest.ckpt')
    assert resumed.global_step == reference.global_step
    assert resumed.cumulative_counts == reference.cumulative_counts
    for role in ('policy', 'ema'):
        for key, tensor in getattr(reference, role).state_dict().items():
            torch.testing.assert_close(tensor, getattr(resumed, role).state_dict()[key], rtol=0, atol=0)
    for index, state in reference.optimizer.state_dict()['state'].items():
        for key, value in state.items():
            torch.testing.assert_close(value, resumed.optimizer.state_dict()['state'][index][key], rtol=0, atol=0)
    assert resumed.scheduler.state_dict() == reference.scheduler.state_dict()


def test_transport_contract_rejects_scientific_change():
    old = {'source_sha256': 'old', 'batch_size': 32, 'ablation': 'C'}
    assert preserve_training_contract({**old, 'source_sha256': 'new'}, old) == old
    with pytest.raises(ValueError, match='beyond data transport'):
        preserve_training_contract({**old, 'batch_size': 8}, old)


@pytest.mark.parametrize('corruption', ['geometry', 'identity', 'total_bytes'])
def test_cache_metadata_rejects_silent_resize_or_wrong_identity(dataset_arguments, tmp_path, corruption):
    import json
    from omegaconf import OmegaConf
    from adavip.manifeel.stage2_data import validate_complete_ram_cache
    source = Stage2SourceDataset(**dataset_arguments)
    cache = tmp_path / 'cache'
    manifest = build_decoded_cache(source, str(cache), workers=2)
    config = tmp_path / 'cache.yaml'
    OmegaConf.save(OmegaConf.create({'dataset': {'_target_':
        'adavip.manifeel.stage2_data.Stage2SourceDataset', **dataset_arguments}}), config)
    validate_complete_ram_cache(source, str(cache), str(config), require_full_ram=False)
    if corruption == 'geometry':
        source.shape_meta['obs']['front']['shape'][1] = 4
    elif corruption == 'identity':
        manifest['arrays'][0]['contract'] = 'wrong'
    else:
        manifest['nbytes'] += 1
    (cache / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='geometry|identity|total bytes'):
        validate_complete_ram_cache(source, str(cache), str(config), require_full_ram=False)
