"""Preserve periodic Stage 1 v2 checkpoints without touching the trainer."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import time
from pathlib import Path

import torch


NATIVE_EPOCHS = frozenset((1, 2, 5, 10, 20, 50, 100, 200, 300, 400))
REQUIRED_KEYS = frozenset((
    "cfg", "model", "normalizer", "optimizer", "scheduler", "rng", "sampler",
    "provenance", "epoch", "global_step", "sample_cursor",
))


def checkpoint_metadata(path: Path) -> dict:
    """Inspect metadata on CPU using mmap without reading tensor storage."""
    payload = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    if not REQUIRED_KEYS.issubset(payload):
        raise ValueError(f"Incomplete training checkpoint: {path}")
    metadata = {name: int(payload[name]) for name in ("epoch", "global_step", "sample_cursor")}
    if any(value < 0 for value in metadata.values()):
        raise ValueError(f"Negative checkpoint cursor: {path}")
    if payload["sampler"]["epoch"] != metadata["epoch"] or \
            payload["sampler"]["consumed"] != metadata["sample_cursor"]:
        raise ValueError(f"Inconsistent sampler cursor: {path}")
    metadata["provenance"] = payload["provenance"]
    return metadata


def archive_latest(run_dir: Path, every: int = 10, previous: tuple | None = None) -> dict | None:
    """Pin an atomic latest file and preserve eligible non-native milestones."""
    if every < 1:
        raise ValueError("Checkpoint interval must be positive")
    latest = run_dir / "checkpoints" / "latest.ckpt"
    try:
        status = latest.stat()
    except FileNotFoundError:
        return None
    identity = (status.st_ino, status.st_size, status.st_mtime_ns)
    if identity == previous:
        return None
    archive_dir = run_dir / "checkpoint_archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    capture = archive_dir / f".capture_{os.getpid()}.ckpt"
    os.link(latest, capture)
    try:
        status = capture.stat()
        identity = (status.st_ino, status.st_size, status.st_mtime_ns)
        metadata = checkpoint_metadata(capture)
        result = {**metadata, "identity": identity, "bytes": status.st_size, "event": "observed"}
        epoch = metadata["epoch"]
        if metadata["sample_cursor"] != 0 or epoch == 0 or epoch % every:
            return result
        if epoch in NATIVE_EPOCHS:
            result["event"] = "native_milestone"
            return result
        destination = latest.parent / f"epoch{epoch}.ckpt"
        try:
            os.link(capture, destination)
        except FileExistsError:
            if checkpoint_metadata(destination) != metadata:
                raise FileExistsError(f"Refusing to overwrite different checkpoint: {destination}")
            result["event"] = "already_preserved"
        else:
            result["event"] = "preserved"
        result["path"] = str(destination)
        return result
    finally:
        capture.unlink()


def main() -> int:
    """Watch immutable latest replacements under a single-instance file lock."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--every", type=int, default=10)
    parser.add_argument("--poll-seconds", type=float, default=1.0)
    parser.add_argument("--final-epoch", type=int, default=400)
    parser.add_argument("--once", action="store_true")
    arguments = parser.parse_args()
    if arguments.every < 1 or arguments.poll_seconds <= 0 or arguments.final_epoch < 1:
        parser.error("Interval, polling time and final epoch must be positive")
    if not arguments.run_dir.is_dir():
        parser.error("Training run directory must already exist")
    archive_dir = arguments.run_dir / "checkpoint_archive"
    archive_dir.mkdir(exist_ok=True)
    stop_signal = None

    def request_stop(received: int, frame: object) -> None:
        nonlocal stop_signal
        stop_signal = received

    for signum in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, request_stop)
    torch.set_num_threads(1)
    with (archive_dir / "watcher.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def record(event: dict) -> None:
            event = {**event, "timestamp": time.time()}
            line = json.dumps(event, sort_keys=True, allow_nan=False)
            with (archive_dir / "events.jsonl").open("a") as stream:
                stream.write(line + "\n")
            print(line, flush=True)

        record({"event": "watcher_started", "every": arguments.every,
                "pid": os.getpid(), "native_epochs": sorted(NATIVE_EPOCHS)})
        previous = None
        while stop_signal is None:
            result = archive_latest(arguments.run_dir, arguments.every, previous)
            if result is not None:
                previous = result["identity"]
                record(result)
                if result["epoch"] >= arguments.final_epoch and result["sample_cursor"] == 0:
                    record({"event": "watcher_completed", "epoch": result["epoch"]})
                    return 0
            if arguments.once:
                return 0
            time.sleep(arguments.poll_seconds)
        record({"event": "watcher_stopped", "signal": signal.Signals(stop_signal).name})
        return 75 if stop_signal == signal.SIGUSR1 else 0


if __name__ == "__main__":
    raise SystemExit(main())
