"""Status must distinguish initialization, real updates and retained completion."""
import json

import pytest

from adavip.advanced.progress import render, stage_status


@pytest.mark.parametrize('name', ['complete.json', 'COMPLETE.json'])
def test_completion_marker_variants(tmp_path, name):
    (tmp_path / name).write_text('{}')
    assert stage_status(tmp_path) == 'completed'
    assert stage_status(tmp_path, labels=True) == 'completed'


def test_contract_and_diagnostic_events_are_not_training(tmp_path):
    (tmp_path / 'training_contract.json').write_text('{}')
    events = tmp_path / 'events.jsonl'
    events.write_text(json.dumps(dict(event='stage2_start', step=0)) + '\n')
    assert stage_status(tmp_path) == 'initializing'
    with events.open('a') as stream:
        stream.write(json.dumps(dict(event='train', step=1)) + '\n{"event":')
    assert stage_status(tmp_path) == 'training'


@pytest.mark.parametrize('name', ['latest.ckpt', 'latest.pt', 'latest.pkl'])
def test_retained_checkpoint_proves_started_training(tmp_path, name):
    (tmp_path / 'checkpoints').mkdir()
    (tmp_path / 'checkpoints' / name).touch()
    assert stage_status(tmp_path) == 'training'


def test_pending_restart_preserves_started_stage_but_not_initialization(tmp_path):
    labels, output = tmp_path / 'labels', tmp_path / 'train'
    labels.mkdir(); output.mkdir()
    (labels / 'COMPLETE.json').write_text('{}')
    (output / 'contract.json').write_text('{}')
    registry = {'dp_stage2_runs': [dict(route='Tabero DP + UViP', job_id='123',
                                      labels=str(labels), output=str(output))]}
    assert '| queued |' in render(registry, {'123': 'PENDING'})
    (output / 'checkpoints').mkdir()
    (output / 'checkpoints/latest.ckpt').touch()
    assert '| training |' in render(registry, {'123': 'PENDING'})
    assert '| failed |' in render(registry, {'123': 'FAILED'})
