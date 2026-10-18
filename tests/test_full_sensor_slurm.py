"""Check safe pending-successor duration changes without contacting Slurm."""

import os
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize("state,matching,parent_state,expected", [
    ("PENDING", True, "COMPLETED", True),
    ("RUNNING", True, "COMPLETED", False),
    ("PENDING", False, "COMPLETED", False),
    ("PENDING", True, "FAILED", False),
])
def test_duration_guard_only_changes_matching_pending_successor(
    tmp_path: Path, state: str, matching: bool, parent_state: str, expected: bool,
) -> None:
    import sys

    root = Path(__file__).resolve().parents[1]
    script = root / "slurm/manifeel/set_full_sensor_successor_duration.sbatch"
    run = tmp_path / "run"
    run.mkdir()
    (run / "continuation_jobs.log").write_text("Submitted batch job 11\nSubmitted batch job 12\n")
    binaries = tmp_path / "bin"
    binaries.mkdir()
    stub = f'''#!{sys.executable}
import os
import sys
from pathlib import Path
args = sys.argv[1:]
if Path(sys.argv[0]).name == "sacct":
    print(os.environ["FAKE_PARENT_STATE"])
elif args[:2] == ["show", "job"]:
    run = os.environ["ADAVIP_RUN_DIR"] if os.environ["FAKE_MATCHING"] == "1" else "/unrelated"
    changed = Path(os.environ["FAKE_UPDATED"]).exists()
    duration = "2-00:00:00" if changed else "06:00:00"
    print(f"JobId=12 JobState={{os.environ['FAKE_STATE']}} TimeLimit={{duration}} TimeMin={{duration}} Command={{run}}/source/slurm/manifeel/train_full_sensor_decoded_batch64.sbatch ")
else:
    with Path(os.environ["FAKE_CALLS"]).open("a") as stream:
        stream.write(" ".join(args) + "\\n")
    if args[0] == "update":
        Path(os.environ["FAKE_UPDATED"]).touch()
'''
    for binary in ("sacct", "scontrol"):
        path = binaries / binary
        path.write_text(stub)
        path.chmod(0o700)
    calls = tmp_path / "calls"
    result = subprocess.run(["bash", str(script)], capture_output=True, text=True, timeout=20, env={
        **os.environ, "PATH": f"{binaries}:{os.environ['PATH']}", "ADAVIP_RUN_DIR": str(run),
        "ADAVIP_PARENT_JOB_ID": "11", "FAKE_STATE": state, "FAKE_MATCHING": str(int(matching)),
        "FAKE_PARENT_STATE": parent_state, "FAKE_CALLS": str(calls), "FAKE_UPDATED": str(tmp_path / "updated"),
    })
    assert (result.returncode == 0) == expected, result.stdout + result.stderr
    if expected:
        assert calls.read_text().splitlines() == ["hold 12", "update JobId=12 TimeLimit=2-00:00:00 TimeMin=2-00:00:00", "release 12"]
        assert (run / "duration_override.log").is_file()
    else:
        assert not calls.exists()


def test_single_gpu_launcher_does_not_allow_shortened_allocations() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "slurm/manifeel/train_full_sensor_decoded_batch64.sbatch").read_text()
    assert "#SBATCH --time=2-00:00:00" in source
    assert "--time-min" not in source
