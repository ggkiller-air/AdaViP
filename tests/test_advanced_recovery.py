"""Scheduler recovery uses durable progress and never duplicates live workloads."""

import json
import os
from pathlib import Path

import pytest

from adavip.advanced import recovery


@pytest.mark.parametrize("record,expected", [
    ({"state": "FAILED", "exit_code": "15:0", "derived_exit_code": "0:15"}, True),
    ({"state": "FAILED", "exit_code": "0:9"}, True),
    ({"state": "FAILED", "exit_code": "1:0"}, False),
    ({"state": "PREEMPTED"}, True),
    ({"state": "TIMEOUT"}, True),
    ({"state": "NODE_FAIL"}, True),
    ({"state": "COMPLETED"}, True),
    ({"state": "CANCELLED"}, False),
    ({"state": "OUT_OF_MEMORY"}, False),
    ({"state": "RUNNING"}, False),
])
def test_retry_infrastructure_interruptions_only(record, expected):
    assert recovery.retryable(record) == expected
    assert not recovery.retryable(record, python_failure=True)


def test_pending_jobs_completion_and_explicit_pause_prevent_retries(tmp_path):
    killed = {"state": "FAILED", "exit_code": "15:0"}
    assert recovery.decide(tmp_path, "1", killed, True, {}, 100)[0] == "active"
    (tmp_path / "RECOVERY_PAUSE").touch()
    assert recovery.decide(tmp_path, "1", killed, False, {}, 100)[0] == "paused"
    (tmp_path / "COMPLETE.json").touch()
    assert recovery.decide(tmp_path, "1", killed, True, {}, 100)[0] == "complete"


def test_terminal_accounting_and_backoff_survive_supervisor_restart(tmp_path):
    killed = {"state": "FAILED", "exit_code": "15:0"}
    action, state = recovery.decide(tmp_path, "1", None, False, {}, 100)
    assert action == "accounting_pending"
    action, state = recovery.decide(tmp_path, "1", killed, False, state, 200)
    assert action == "backoff" and state["retry_after"] == 220
    restored = json.loads(json.dumps(state))
    assert recovery.decide(tmp_path, "1", killed, False, restored, 220)[0] == "retry"


def test_current_python_exception_stops_but_old_failure_does_not(tmp_path):
    killed = {"state": "FAILED", "exit_code": "15:0"}
    path = tmp_path / "FAILED.json"
    path.write_text(json.dumps({"job_id": "1", "error": "invalid label identity"}))
    assert recovery.decide(tmp_path, "1", killed, False, {}, 100)[0] == "stopped"
    path.write_text(json.dumps({"job_id": "older"}))
    assert recovery.decide(tmp_path, "1", killed, False, {}, 100)[0] == "backoff"


def test_retry_backoff_is_bounded_and_resets_on_checkpoint_progress(tmp_path):
    killed = {"state": "FAILED", "exit_code": "15:0"}
    state = {"observed_job": "1", "terminal_since": 0,
             "no_progress_retries": 7, "progress_token": recovery.progress_token(tmp_path)}
    action, state = recovery.decide(tmp_path, "1", killed, False, state, 2000)
    assert action == "retry" and state["retry_after"] == 1800
    state["no_progress_retries"] = 8
    assert recovery.decide(tmp_path, "1", killed, False, state, 2000)[0] == "no_progress_limit"
    checkpoint = tmp_path / "checkpoints/latest.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"new durable checkpoint")
    action, state = recovery.decide(tmp_path, "1", killed, False, state, 2000)
    assert action == "retry" and state["no_progress_retries"] == 0


def fixture_registry(tmp_path):
    output = tmp_path / "task"
    output.mkdir()
    config = output / "launch_config.json"
    config.write_text(json.dumps({"task": "insert_hole", "source_id": "unchanged"}))
    run = {"output": str(output), "config": str(config), "model": "ftp1",
           "source": "/immutable/original", "job_id": "1", "account": "overload"}
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps({"runs": [run], "source": "/default"}))
    return registry, run


def test_continuation_is_adopted_only_when_output_matches(tmp_path, monkeypatch):
    registry, run = fixture_registry(tmp_path)
    queue = [{"job": "2", "name": "uvip_ftp1_insert_hole", "state": "PENDING"}]
    monkeypatch.setattr(recovery, "slurm", lambda command: f"StdOut={run['output']}/slurm_2.out")
    assert recovery.active_job(run, queue) == "2"
    monkeypatch.setattr(recovery, "slurm", lambda command: "StdOut=/another/experiment/slurm_2.out")
    assert recovery.active_job(run, queue) is None


def test_poll_resumes_original_source_and_cannot_double_submit(tmp_path, monkeypatch):
    registry, run = fixture_registry(tmp_path)
    config_before = Path(run["config"]).read_bytes()
    queue = []
    submissions = []
    monkeypatch.setattr(recovery, "queue_snapshot", lambda: queue)
    monkeypatch.setattr(recovery, "accounting", lambda jobs: {
        "1": {"state": "FAILED", "exit_code": "15:0"}})
    monkeypatch.setattr(recovery.time, "time", lambda: 1000)

    def submit(source, output, config, model):
        submissions.append((source, str(output), config, model))
        assert os.environ["UVIP_SLURM_ACCOUNT"] == "overload"
        assert os.environ["UVIP_RUN_REGISTRY"] == str(registry)
        recovery.register_job(registry, Path(output), "2")
        queue.append({"job": "2", "name": "uvip_ftp1_insert_hole", "state": "PENDING"})
        return "2"

    monkeypatch.setattr(recovery, "submit", submit)
    monkeypatch.setenv("UVIP_SLURM_ACCOUNT", "original")
    monkeypatch.setenv("UVIP_RUN_REGISTRY", "original")
    recovery.poll(registry, grace=0)
    # A real grace window also applies when grace=0; advance to the next poll.
    monkeypatch.setattr(recovery.time, "time", lambda: 1060)
    recovery.poll(registry, grace=0)
    recovery.poll(registry, grace=0)
    assert len(submissions) == 1
    assert submissions[0][0] == "/immutable/original"
    assert Path(run["config"]).read_bytes() == config_before
    assert json.loads(registry.read_text())["runs"][0]["job_id"] == "2"


def test_queue_is_checked_again_before_submission(tmp_path, monkeypatch):
    registry, run = fixture_registry(tmp_path)
    count = 0

    def queue():
        nonlocal count
        count += 1
        return [] if count < 3 else [{"job": "2", "name": "uvip_ftp1_insert_hole", "state": "PENDING"}]

    monkeypatch.setattr(recovery, "queue_snapshot", queue)
    monkeypatch.setattr(recovery, "accounting", lambda jobs: {"1": {"state": "TIMEOUT"}})
    monkeypatch.setattr(recovery, "slurm", lambda command: f"StdOut={run['output']}/slurm_2.out")
    monkeypatch.setattr(recovery.time, "time", lambda: 1000)
    recovery.poll(registry)
    monkeypatch.setattr(recovery.time, "time", lambda: 1120)
    monkeypatch.setattr(recovery, "submit", lambda *args: pytest.fail("duplicate submission"))
    recovery.poll(registry)
    assert json.loads(registry.read_text())["runs"][0]["job_id"] == "2"


def test_scheduler_outage_fails_closed(tmp_path, monkeypatch):
    registry, run = fixture_registry(tmp_path)

    def unavailable():
        raise TimeoutError("Slurm unavailable")

    monkeypatch.setattr(recovery, "queue_snapshot", unavailable)
    monkeypatch.setattr(recovery, "submit", lambda *args: pytest.fail("scheduler outage caused a submission"))
    with pytest.raises(TimeoutError):
        recovery.poll(registry)
