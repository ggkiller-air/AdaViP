"""Atomic artifacts, experiment identities and automatic A/B/C scheduling."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def append_event(path, event, **fields):
    value = {"time": time.strftime("%Y-%m-%d %H:%M:%S %z"), "event": event, **fields}
    with open(path, "a") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")
    print(json.dumps(value, ensure_ascii=False), flush=True)
    if event in {"phase","labels_complete","train_complete","stage2_start"} or (event=="labels" and fields.get("done",0)<=32):
        try:
            from .progress import update
            update()
        except Exception as error:
            print(f"Progress Markdown update failed: {error}",flush=True)


def presence_table(model):
    size = 6 if model == "ftp1" else 5
    extra = 2 if model == "ftp1" else 1
    values = np.ones((1 + size + extra, size), bool)
    for index in range(size):
        values[index + 1, index] = False
    values[size + 1, :2] = False
    if extra == 2:
        values[-1, 2:4] = False
    return values


def phase_at(step, config):
    if step < config["phase_a_steps"]:
        return "A"
    if step < config["phase_a_steps"] + config["phase_b_steps"]:
        return "B"
    return "C"


def sampled_indices(length, batch_size, seed, step):
    # Sampling depends on optimizer step, independent of worker prefetch/resume.
    return np.random.default_rng(np.random.SeedSequence([seed, step])).integers(0, length, batch_size)


def noise_draw(ids, horizon, dimension, draw, seed=42):
    return np.stack([np.random.default_rng(np.random.SeedSequence([seed, int(i), draw])).standard_normal(
        (horizon, dimension), dtype=np.float32) for i in ids])


def quadrature_time(draw, count):
    # Inverse CDF of native Beta(1.5,1), stratified midpoints.
    return .001 + .999 * ((draw + .5) / count) ** (2 / 3)


def stop_requested():
    path = os.environ.get("UVIP_STOP_FILE")
    return bool(path and Path(path).exists())
