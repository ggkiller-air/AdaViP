"""Exercise the real JAX A/B/C trainer and interrupted resume on real labels.

This diagnostic never supplies a formal Student checkpoint. All updated
adapter parameters are restored before the fixed Teacher continues scoring.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

from adavip.tabero.vtla import trainable_filter
from .common import atomic_json, fingerprint


def tabero(model,data,config,output,utility):
    from . import tabero_run as runner

    root = output / "trainer_smoke"
    if (root / "COMPLETE.json").exists():
        return
    root.mkdir(exist_ok=True)
    label_dir = root / "labels"
    label_dir.mkdir(exist_ok=True)
    with open(label_dir / "utility.npy","wb") as handle:
        np.save(handle,utility)
    rms = np.maximum(np.sqrt(np.mean(utility.astype(np.float64)**2,0)),1e-4)
    atomic_json(label_dir / "COMPLETE.json",{"identity":fingerprint({"diagnostic_first_windows":len(utility),"source":config["source_id"]}),
                "rms":rms.tolist(),"windows":len(utility)})
    class SmallData:
        def __len__(self):
            return len(utility)
        def batch(self,ids):
            return data.batch(ids)
    small = SmallData()
    settings = copy.deepcopy(config)
    settings.update(phase_a_steps=2,phase_b_steps=2,phase_c_steps=2,batch_size=4,micro_batch_size=4,
                    save_every=1,periodic_every=1,log_every=1,archive=str(root / "archive"))
    selected = trainable_filter("B")
    initial = jax.device_get(nnx.state(model,selected))
    enabled = model.uvip_enabled
    original_stop = runner.stop_requested
    try:
        model.uvip_enabled = True
        split_dir,reference_dir = root / "interrupted",root / "uninterrupted"
        split_dir.mkdir(exist_ok=True)
        reference_dir.mkdir(exist_ok=True)
        calls = [0]
        def stop_after_three():
            calls[0] += 1
            return original_stop() or calls[0]==3
        runner.stop_requested = stop_after_three
        runner.train(model,small,settings,split_dir,label_dir)
        runner.stop_requested = original_stop
        if original_stop():
            return
        # Reconstruct inherited+adapter state; actual trainer restores its own
        # delta/Adam/cursor and deterministic step-based PRNG from checkpoint.
        nnx.update(model,jax.tree.map(jnp.asarray,initial))
        runner.train(model,small,settings,split_dir,label_dir)
        resumed = jax.device_get(nnx.state(model,selected))
        nnx.update(model,jax.tree.map(jnp.asarray,initial))
        runner.train(model,small,settings,reference_dir,label_dir)
        reference = jax.device_get(nnx.state(model,selected))
        for left,right in zip(jax.tree.leaves(resumed),jax.tree.leaves(reference)):
            np.testing.assert_array_equal(left,right)
        report = {"kind":"diagnostic only; not formal Stage2", "phases":["A","B","C"],
                  "optimizer_updates":6,"interruption_after":3,
                  "resume_vs_uninterrupted":"exact all selected parameters",
                  "teacher_adapter_restored":True}
        atomic_json(root / "COMPLETE.json",report)
        print("PASS real A/B/C trainer and exact interrupted resume",flush=True)
    finally:
        runner.stop_requested = original_stop
        nnx.update(model,jax.tree.map(jnp.asarray,initial))
        model.uvip_enabled = enabled
