"""Native JAX Teacher labels and one continuous A/B/C action-LoRA Student."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path
import pickle
import time

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from .common import (append_event, atomic_json, file_hash, fingerprint, noise_draw,
                     phase_at, presence_table, quadrature_time, sampled_indices, stop_requested)
from .data import TaberoData
from adavip.tabero.vtla import load_released, trainable_filter


def generate_labels(model,data,config,output):
    directory = output / "labels"
    directory.mkdir(parents=True,exist_ok=True)
    contract = {"model":"native JAX Pi0 Field+FS weighted_full", "data":data.contract,
                "teacher":config["teacher"],"teacher_metadata_sha256":file_hash(Path(config["teacher"]) / "_METADATA"),
                "teacher_revision":config["teacher_revision"],"k":config["utility_k"],
                "noise_seed":config["seed"],"time":"Beta(1.5,1) stratified inverse-CDF midpoints*.999+.001",
                "presence":presence_table("tabero").tolist(),"signed":"masked_loss-full_loss",
                "implementation":config["source_id"],"forward":"native full prefix+suffix; cached prefusion embeddings only"}
    identity = fingerprint(contract)
    if (directory / "COMPLETE.json").exists():
        if json.loads((directory / "COMPLETE.json").read_text())["identity"] != identity:
            raise ValueError("Label identity changed")
        return directory
    if (directory / "contract.json").exists() and json.loads((directory / "contract.json").read_text()) != contract:
        raise ValueError("Cannot resume labels with changed contract")
    atomic_json(directory / "contract.json",contract)
    model.uvip_enabled = False
    encode_fn = nnx.jit(lambda m,o,p:m.encode_observation(o,p)[0])
    score_fn = nnx.jit(lambda m,o,a,p,n,t,e:m.flow_loss(jax.random.key(0),o,a,presence=p,noise=n,time=t,encoded_prefix=e))
    table = presence_table("tabero")
    count,batch_size = config["utility_k"],config["score_batch_size"]
    weights = jnp.ones(32,jnp.float32).at[7:13].set(.1)
    started = time.monotonic()
    for start in range(0,len(data),batch_size):
        ids = np.arange(start,min(start+batch_size,len(data)))
        # Keep compiled shapes constant at the final batch; discard duplicates.
        padded_ids = np.pad(ids,(0,batch_size-len(ids)),mode="edge")
        path = directory / f"batch_{start:07d}.npz"
        if path.exists():
            with np.load(path) as previous:
                if not np.array_equal(previous["ids"],ids) or str(previous["identity"]) != identity:
                    raise ValueError(f"Invalid committed label shard {path}")
            continue
        obs,actions = data.batch(padded_ids)
        losses = np.zeros((len(ids),len(table)),np.float64)
        components = np.zeros((len(ids),len(table),3),np.float64)
        for index,flags in enumerate(table):
            presence = jnp.asarray(np.broadcast_to(flags,(batch_size,5)).copy())
            encoded = encode_fn(model,obs,presence)
            for draw in range(count):
                noise = jnp.asarray(noise_draw(padded_ids,50,32,draw,config["seed"]))
                times = jnp.full((batch_size,),quadrature_time(draw,count))
                result = score_fn(model,obs,actions,presence,noise,times,encoded)
                scores = result["per_sample_loss"]
                part = jnp.stack([result["position_loss"],result["force_loss"],result["padding_loss"]],-1)
                losses[:,index] += np.asarray(scores)[:len(ids)]/count
                components[:,index] += np.asarray(part)[:len(ids)]/count
            del encoded
        utility = (losses[:,1:]-losses[:,:1]).astype(np.float32)
        if not np.isfinite(losses).all():
            raise FloatingPointError("Nonfinite Teacher score")
        temporary = path.with_suffix(".tmp")
        with open(temporary,"wb") as handle:
            np.savez(handle,ids=ids,utility=utility,losses=losses.astype(np.float32),
                     components=components.astype(np.float32),identity=identity)
        os.replace(temporary,path)
        if start==0:
            from .trainer_smoke import tabero as smoke_trainer
            smoke_trainer(model,data,config,output,utility)
        if start==0 or (start//batch_size)%20==0 or ids[-1]+1==len(data):
            append_event(output / "events.jsonl","labels",done=int(ids[-1]+1),total=len(data),
                         elapsed_seconds=time.monotonic()-started)
        if stop_requested():
            return None
    utility = np.empty((len(data),6),np.float32)
    for start in range(0,len(data),batch_size):
        with np.load(directory / f"batch_{start:07d}.npz") as shard:
            utility[shard["ids"]] = shard["utility"]
    rms = np.maximum(np.sqrt(np.mean(utility.astype(np.float64)**2,0)),1e-4).astype(np.float32)
    with open(directory / "utility.tmp","wb") as handle:
        np.save(handle,utility)
    os.replace(directory / "utility.tmp",directory / "utility.npy")
    atomic_json(directory / "COMPLETE.json",{"identity":identity,"windows":len(data),"k":count,
                "rms":rms.tolist(),"utility_sha256":file_hash(directory / "utility.npy"),
                "negative_fraction":(utility<0).mean(0).tolist(),"mean":utility.mean(0).tolist()})
    append_event(output / "events.jsonl","labels_complete",windows=len(data),identity=identity)
    model.uvip_enabled = True
    return directory


def train(model,data,config,output,label_directory):
    manifest = json.loads((label_directory / "COMPLETE.json").read_text())
    labels = np.load(label_directory / "utility.npy",mmap_mode="r")
    rms = jnp.asarray(manifest["rms"])
    identity = fingerprint({"config":config,"label_identity":manifest["identity"]})
    selected_filter = trainable_filter("B")
    params = nnx.state(model,selected_filter)
    tx = optax.chain(optax.clip_by_global_norm(1.),optax.scale_by_adam(b1=.9,b2=.95,eps=1e-8),
                     optax.add_decayed_weights(1e-10),optax.scale(-1.))
    optimizer = tx.init(params)
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    latest = checkpoints / "latest.pkl"
    step = 0
    if latest.exists():
        with open(latest,"rb") as handle:
            saved = pickle.load(handle)
        if saved["identity"] != identity:
            raise ValueError("Student checkpoint identity mismatch")
        nnx.update(model,jax.tree.map(jnp.asarray,saved["delta"]))
        optimizer = jax.tree.map(jnp.asarray,saved["optimizer"])
        step = saved["step"]
    total = sum(config[key] for key in ("phase_a_steps","phase_b_steps","phase_c_steps"))
    atomic_json(output / "training_contract.json",{"identity":identity,"labels":manifest,"config":config})
    append_event(output / "events.jsonl","stage2_start",step=step,total=total,identity=identity)
    audit = {"/".join(map(str,p)):list(v.value.shape) for p,v in params.flat_state().items()}
    atomic_json(output / "trainable_parameters.json",audit)
    def loss_a(m,key,obs,actions,target):
        utility = jnp.mean((m.utility(obs)-target)**2)
        return utility,{"utility_loss":utility,"action_loss":jnp.zeros(()),"position_loss":jnp.zeros(()),"force_loss":jnp.zeros(())}
    def loss_bc(m,key,obs,actions,target):
        result = m.flow_loss(key,obs,actions)
        utility = jnp.mean((result["utility"]-target)**2)
        action = jnp.mean(result["loss"])
        return action+config["utility_loss_weight"]*utility,{
            "utility_loss":utility,"action_loss":action,
            "position_loss":result["position_loss"].mean(),"force_loss":result["force_loss"].mean()}
    grad_a = nnx.jit(nnx.value_and_grad(loss_a,argnums=nnx.DiffState(0,selected_filter),has_aux=True))
    grad_bc = nnx.jit(nnx.value_and_grad(loss_bc,argnums=nnx.DiffState(0,selected_filter),has_aux=True))
    def role(path):
        name = "/".join(map(str,path))
        return "utility" if name.startswith("uvip/utility_predictor/") else "uvip" if name.startswith("uvip/") else "action_lora"
    @nnx.jit
    def apply(m,opt_state,grads,scale,only_utility):
        values = nnx.state(m,selected_filter)
        updates,new_state = tx.update(grads,opt_state,values)
        updates = updates.map(lambda path,var:var.replace(var.value*scale*config["learning_rates"][role(path)]*
                                              jnp.where(only_utility & (role(path)!="utility"),0.,1.)))
        nnx.update(m,optax.apply_updates(values,updates))
        return new_state
    total = sum(config[key] for key in ("phase_a_steps","phase_b_steps","phase_c_steps"))
    def save(periodic=False):
        saved = {"identity":identity,"step":step,"phase":phase_at(min(step,total-1),config),
                 "delta":jax.device_get(nnx.state(model,selected_filter)),
                 "optimizer":jax.device_get(optimizer),"seed":config["seed"],
                 "rng_scheme":"fold_in(PRNGKey(seed),step*batch_size+micro_offset)",
                 "teacher":config["teacher"],"label_identity":manifest["identity"],"config":config}
        with open(checkpoints / "latest.tmp","wb") as handle:
            pickle.dump(saved,handle,protocol=5)
        os.replace(checkpoints / "latest.tmp",latest)
        if periodic:
            path = checkpoints / f"step_{step:07d}.pkl"
            if not path.exists():
                os.link(latest,path)
        if step==1:
            archive = Path(config["archive"])
            archive.mkdir(parents=True,exist_ok=True)
            if not (archive / "initial_student.pkl").exists():
                os.link(latest,archive / "initial_student.pkl")
    previous_phase = None
    window_start,window_step = time.monotonic(),step
    micro = config["micro_batch_size"]
    while step<total:
        phase = phase_at(step,config)
        if phase!=previous_phase:
            append_event(output / "events.jsonl","phase",phase=phase,step=step,total=total)
            previous_phase = phase
        phase_start = 0 if phase=="A" else config["phase_a_steps"] if phase=="B" else config["phase_a_steps"]+config["phase_b_steps"]
        local_step,duration = step-phase_start,config[f"phase_{phase.lower()}_steps"]
        warmup = min(200 if phase=="A" else 1000,duration//5)
        scale = min(1.,(local_step+1)/max(warmup,1))*(.55+.45*math.cos(math.pi*local_step/max(duration,1)))
        if phase=="C":
            scale *= .1
        ids = sampled_indices(len(data),config["batch_size"],config["seed"],step)
        grads,metrics = None,None
        for offset in range(0,len(ids),micro):
            selected = ids[offset:offset+micro]
            obs,actions = data.batch(selected)
            target = jnp.asarray(np.asarray(labels[selected]))/rms
            key = jax.random.fold_in(jax.random.key(config["seed"]),step*config["batch_size"]+offset)
            (loss,aux),gradient = (grad_a if phase=="A" else grad_bc)(model,key,obs,actions,target)
            if not np.isfinite(float(loss)):
                raise FloatingPointError(f"Nonfinite training loss at {step}")
            weight = len(selected)/len(ids)
            grads = jax.tree.map(lambda value:value*weight,gradient) if grads is None else jax.tree.map(
                lambda old,new:old+weight*new,grads,gradient)
            metrics = jax.tree.map(lambda value:value*weight,aux) if metrics is None else jax.tree.map(
                lambda old,new:old+weight*new,metrics,aux)
        norm = optax.global_norm(grads)
        if not np.isfinite(float(norm)):
            raise FloatingPointError(f"Nonfinite gradient at {step}")
        optimizer = apply(model,optimizer,grads,jnp.asarray(scale),jnp.asarray(phase=="A"))
        step += 1
        if step==1 or step%config["log_every"]==0:
            jax.block_until_ready(optimizer)
            elapsed = time.monotonic()-window_start
            append_event(output / "events.jsonl","train",step=step,total=total,phase=phase,
                         **{k:float(v) for k,v in metrics.items()},grad_norm=float(norm),
                         samples_per_second=(step-window_step)*config["batch_size"]/elapsed,
                         learning_rate_scale=scale)
            window_start,window_step = time.monotonic(),step
        if step==1 or step%config["save_every"]==0 or step==total or stop_requested():
            save(periodic=(step==1 or step%config["periodic_every"]==0 or step==total))
        if phase=="B" and local_step==99:
            atomic_json(output / "B_100_updates.json",{"step":step,"batch_size":config["batch_size"],
                                                       "micro_batch_size":micro,"finite":True})
        if stop_requested():
            return
    atomic_json(output / "COMPLETE.json",{"identity":identity,"step":step,"checkpoint":str(latest),
                                          "label_identity":manifest["identity"]})
    append_event(output / "events.jsonl","train_complete",step=step,checkpoint=str(latest))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",required=True)
    parser.add_argument("--output",required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    output = Path(args.output)
    output.mkdir(parents=True,exist_ok=True)
    if (output / "COMPLETE.json").exists():
        return
    preflight = output / "preflight"
    if not (preflight / "COMPLETE.json").exists():
        from .preflight import tabero
        preflight.mkdir(exist_ok=True)
        tabero(config,preflight)
        gc.collect()
        jax.clear_caches()
    if stop_requested():
        return
    data = TaberoData(config["cache"],config["normalizer"])
    model,_ = load_released(config["teacher"],seed=config["seed"])
    labels = generate_labels(model,data,config,output)
    if labels is not None and not stop_requested():
        model.uvip_enabled = True
        # Discard scoring compilations before compiling gradient paths.
        jax.clear_caches()
        train(model,data,config,output,labels)


if __name__=="__main__":
    main()
