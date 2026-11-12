"""Archive and publish only this candidate's immutable protocol checkpoints."""

from __future__ import annotations

import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import time


EPOCHS = (1, 6, 10, 20, 50, 100, 200, 300, 400, 500, 600)
PREFIX = "ManiFeel/ablation/C_teacher_residual"
REPO = "ggkiller/UVIP"


def main() -> None:
    """Use a CPU-only process; never read a partial/latest file or alter training."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--prefix", default=PREFIX,
                        help="Unique remote experiment prefix; defaults to the original B32 run")
    args = parser.parse_args()
    prefix = args.prefix.rstrip('/')
    if not prefix.startswith('ManiFeel/ablation/') or '..' in prefix.split('/'):
        raise ValueError('Unexpected publication prefix')
    root = args.root.resolve()
    run = root / "C_teacher_residual"
    work = root / "publication"
    work.mkdir(exist_ok=True)
    archive = Path("/data/wangzihao/archives/manifeel_uvip") / root.name / "published"
    archive.mkdir(parents=True, exist_ok=True)
    for key in list(os.environ):
        if key.lower().endswith("_proxy"):
            del os.environ[key]
    # SDK authentication is read from the user's existing credential store.
    from modelscope_hub.api import HubApi

    lock = (work / "publisher.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    stopped = False

    def stop(*_):
        nonlocal stopped
        stopped = True

    for signum in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, stop)

    def emit(event, **values):
        print(json.dumps(dict(time=datetime.datetime.now().astimezone().isoformat(),
                              event=event, **values)), flush=True)

    while not stopped:
        try:
            registry = json.loads((root / "registry.json").read_text())
            smoke = Path(registry["smoke_output"])
            assert json.loads((smoke / "SMOKE_PASS.json").read_text())["status"] == "PASS"
            source = Path(registry["source"])
            api = HubApi()
            remote = {f.path: f.size for f in api.list_repo_files(REPO, "model", revision="master")
                      if not f.is_dir and f.path.startswith(prefix + "/")}
            files = [(work / "README.md", "README.md"),
                     (source / "source.sha256", "source.sha256"),
                     (source / "source_provenance.json", "source_provenance.json"),
                     (smoke / "SMOKE_PASS.json", "SMOKE_PASS.json"),
                     (smoke / "teacher_parity.json", "teacher_parity.json")]
            files += [(run / name, name) for name in
                      ("inputs_verified.json", "initialization_audit.json", "architecture.json", "contract.json")]
            files += [(run / "checkpoints" / f"epoch{epoch}.ckpt", f"epoch{epoch}.ckpt") for epoch in EPOCHS]
            for path, name in files:
                if stopped:
                    break
                if not path.is_file():
                    continue
                destination = prefix + "/" + name
                size = path.stat().st_size
                if destination in remote:
                    if remote[destination] != size:
                        raise RuntimeError("Refusing to overwrite published artifact: " + destination)
                    continue
                if name.endswith(".ckpt"):
                    archived = archive / name
                    if not archived.exists():
                        temporary = archive / (name + ".partial")
                        shutil.copyfile(path, temporary)
                        assert temporary.stat().st_size == size
                        os.replace(temporary, archived)
                    assert archived.stat().st_size == size
                    path = archived
                emit("upload_start", path=destination, bytes=size)
                api.upload_file(REPO, "model", path, destination, revision="master",
                                commit_message="Publish C_teacher_residual " + name, disable_tqdm=True)
                confirmed = {f.path: f.size for f in api.list_repo_files(REPO, "model", revision="master")
                             if not f.is_dir}
                assert confirmed[destination] == size
                remote[destination] = size
                emit("upload_verified", path=destination, bytes=size)
            receipt = dict(repo=REPO, prefix=prefix, files=remote,
                           checked_at=datetime.datetime.now().astimezone().isoformat())
            tmp = work / "published.json.partial"
            tmp.write_text(json.dumps(receipt, indent=2) + "\n")
            os.replace(tmp, work / "published.json")
            if (run / "COMPLETE.json").exists() and all(prefix + f"/epoch{e}.ckpt" in remote for e in EPOCHS):
                (work / "COMPLETE.json").write_text(json.dumps(receipt, indent=2) + "\n")
                return
            if args.once:
                return
        except Exception as error:
            emit("retry", error_type=type(error).__name__)
            if args.once:
                raise
        for _ in range(600):
            if stopped:
                break
            time.sleep(1)
    raise SystemExit(75)


if __name__ == "__main__":
    main()
