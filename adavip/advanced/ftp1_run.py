"""One task: verified fixed Teacher -> paired labels -> automatic A/B/C Student."""

from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from .common import (append_event, atomic_json, file_hash, fingerprint, noise_draw,
                     phase_at, presence_table, quadrature_time, sampled_indices, stop_requested)
from .data import FTPData
from adavip.ftp1.integration import load_released
from adavip.ftp1.modulation import per_sample_flow_loss


def generate_labels(model, data, config, output):
    directory = output / "labels"
    directory.mkdir(parents=True, exist_ok=True)
    contract = {"model": "FTP-1 native flow", "data": data.contract,
                "teacher_sha256": config["teacher_sha256"], "k": config["utility_k"],
                "noise_seed": config["seed"], "time": "Beta(1.5,1) stratified inverse-CDF midpoints*.999+.001",
                "presence": presence_table("ftp1").tolist(), "signed": "masked_loss-full_loss",
                "implementation": config["source_id"],"forward":"native full attention; cached prefusion encoder tokens only"}
    identity = fingerprint(contract)
    if (directory / "COMPLETE.json").exists():
        manifest = json.loads((directory / "COMPLETE.json").read_text())
        if manifest["identity"] != identity:
            raise ValueError("Existing labels have another identity")
        return directory
    if (directory / "contract.json").exists():
        if json.loads((directory / "contract.json").read_text()) != contract:
            raise ValueError("Refusing label resume with a changed contract")
    atomic_json(directory / "contract.json", contract)
    model.uvip_enabled = False
    table = torch.tensor(presence_table("ftp1"), device="cuda")
    batch_size, count = config["score_batch_size"], config["utility_k"]
    started = time.monotonic()
    with torch.no_grad():
        for start in range(0, len(data), batch_size):
            end = min(start + batch_size, len(data))
            path = directory / f"batch_{start:07d}.npz"
            if path.exists():
                with np.load(path) as previous:
                    if not np.array_equal(previous["ids"], np.arange(start,end)) or str(previous["identity"]) != identity:
                        raise ValueError(f"Invalid committed label shard {path}")
                continue
            ids = np.arange(start, end)
            obs, actions = data.batch(ids)
            model._raw_features = model.encode_native(obs)
            # Native full attention preserves the exact Teacher objective;
            # only frozen prefusion encoder features are reused across draws.
            losses = np.zeros((len(ids), len(table)), np.float64)
            for index, flags in enumerate(table):
                model.presence = flags[None].expand(len(ids), -1)
                for draw in range(count):
                    noise = torch.from_numpy(noise_draw(ids,32,120,draw,config["seed"])).to("cuda")
                    times = torch.full((len(ids),),quadrature_time(draw,count),device="cuda")
                    result = model(obs,actions,noise=noise,time=times)
                    losses[:,index] += result["per_sample_loss"].double().cpu().numpy()/count
            model._raw_features = None
            utility = (losses[:,1:]-losses[:,:1]).astype(np.float32)
            if not np.isfinite(losses).all() or not np.isfinite(utility).all():
                raise FloatingPointError("Nonfinite native Teacher labels")
            temporary = path.with_suffix(".tmp")
            with open(temporary,"wb") as handle:
                np.savez(handle,ids=ids,utility=utility,losses=losses.astype(np.float32),identity=identity)
            os.replace(temporary,path)
            if start == 0 or end % (batch_size*20) == 0 or end == len(data):
                append_event(output / "events.jsonl","labels",done=end,total=len(data),
                             elapsed_seconds=time.monotonic()-started)
            if stop_requested():
                return None
    utility = np.empty((len(data),8),np.float32)
    for start in range(0,len(data),batch_size):
        with np.load(directory / f"batch_{start:07d}.npz") as shard:
            utility[shard["ids"]] = shard["utility"]
    rms = np.maximum(np.sqrt(np.mean(utility.astype(np.float64)**2,axis=0)),1e-4).astype(np.float32)
    temporary = directory / "utility.tmp"
    with open(temporary,"wb") as handle:
        np.save(handle,utility)
    os.replace(temporary,directory / "utility.npy")
    atomic_json(directory / "COMPLETE.json", {"identity":identity,"windows":len(data),"k":count,
                "rms":rms.tolist(),"utility_sha256":file_hash(directory / "utility.npy"),
                "negative_fraction":(utility<0).mean(0).tolist(),"mean":utility.mean(0).tolist()})
    append_event(output / "events.jsonl","labels_complete",windows=len(data),identity=identity)
    model.presence, model.uvip_enabled = None, True
    return directory


def train(model,data,config,output,label_directory):
    labels = np.load(label_directory / "utility.npy",mmap_mode="r")
    label_manifest = json.loads((label_directory / "COMPLETE.json").read_text())
    rms = torch.tensor(label_manifest["rms"],device="cuda")
    identity = fingerprint({"config":config,"label_identity":label_manifest["identity"]})
    model.set_phase("B")
    params = {n:p for n,p in model.named_parameters() if p.requires_grad}
    role = lambda name: ("utility" if name.startswith("uvip.utility_predictor.") else
                          "action_lora" if not name.startswith("uvip.") else "uvip")
    groups = [{"params":[p for n,p in params.items() if role(n)==key],"role":key,
               "lr":config["learning_rates"][key]} for key in ("utility","uvip","action_lora")]
    optimizer = torch.optim.AdamW(groups,betas=(.9,.95),eps=1e-8,weight_decay=1e-10)
    ema = {n:p.detach().clone() for n,p in params.items()}
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    latest = checkpoints / "latest.pt"
    step = 0
    if latest.exists():
        saved = torch.load(latest,map_location="cpu",weights_only=False)
        if saved["identity"] != identity:
            raise ValueError("Student resume identity mismatch")
        with torch.no_grad():
            for n,p in params.items():
                p.copy_(saved["delta"][n].to(p.device))
        optimizer.load_state_dict(saved["optimizer"])
        ema = {n:v.to("cuda") for n,v in saved["ema"].items()}
        step = saved["step"]
        torch.set_rng_state(saved["torch_rng"])
        torch.cuda.set_rng_state_all(saved["cuda_rng"])
        np.random.set_state(saved["numpy_rng"])
        random.setstate(saved["python_rng"])
    total = sum(config[key] for key in ("phase_a_steps","phase_b_steps","phase_c_steps"))
    atomic_json(output / "training_contract.json",{"identity":identity,"labels":label_manifest,"config":config})
    append_event(output / "events.jsonl","stage2_start",step=step,total=total,identity=identity)
    atomic_json(output / "trainable_parameters.json",{n:list(p.shape) for n,p in params.items()})
    def save(periodic=False):
        saved = {"identity":identity,"step":step,"phase":phase_at(min(step,total-1),config),
                 "delta":{n:p.detach().cpu() for n,p in params.items()},
                 "ema":{n:v.cpu() for n,v in ema.items()},"optimizer":optimizer.state_dict(),
                 "torch_rng":torch.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state_all(),
                 "numpy_rng":np.random.get_state(),"python_rng":random.getstate(),
                 "teacher":config["teacher"],"label_identity":label_manifest["identity"],"config":config}
        temporary = checkpoints / "latest.tmp"
        torch.save(saved,temporary)
        os.replace(temporary,latest)
        if periodic:
            periodic_path = checkpoints / f"step_{step:07d}.pt"
            if not periodic_path.exists():
                os.link(latest,periodic_path)
        if step == 1:
            archive = Path(config["archive"])
            archive.mkdir(parents=True,exist_ok=True)
            if not (archive / "initial_student.pt").exists():
                os.link(latest,archive / "initial_student.pt")
    previous_phase = None
    window_start, window_step = time.monotonic(),step
    while step < total:
        phase = phase_at(step,config)
        if phase != previous_phase:
            model.set_phase(phase)
            append_event(output / "events.jsonl","phase",phase=phase,step=step,total=total)
            previous_phase = phase
        phase_start = 0 if phase=="A" else config["phase_a_steps"] if phase=="B" else config["phase_a_steps"]+config["phase_b_steps"]
        duration = config[f"phase_{phase.lower()}_steps"]
        local_step = step-phase_start
        warmup = min(100 if phase=="A" else 500,duration//5)
        scale = min(1.,(local_step+1)/max(warmup,1))*(.55+.45*math.cos(math.pi*local_step/max(duration,1)))
        if phase=="C":
            scale *= .1
        for group in optimizer.param_groups:
            group["lr"] = config["learning_rates"][group["role"]]*scale
        ids = sampled_indices(len(data),config["batch_size"],config["seed"],step)
        model.zero_grad(set_to_none=True)
        utility_metric, action_metric = 0.,0.
        micro = config["micro_batch_size"]
        for offset in range(0,len(ids),micro):
            selected = ids[offset:offset+micro]
            obs,actions = data.batch(selected)
            target = torch.as_tensor(np.asarray(labels[selected]),device="cuda")/rms
            weight = len(selected)/len(ids)
            if phase=="A":
                model._preprocess_observation(obs,train=False)
                pred = model.last_utility
                model._prepared = None
                uloss = torch.nn.functional.mse_loss(pred,target)
                aloss = torch.zeros((),device="cuda")
                loss = uloss
            else:
                result = model(obs,actions)
                uloss = torch.nn.functional.mse_loss(result["utility"],target)
                aloss = result["loss"]
                loss = aloss+config["utility_loss_weight"]*uloss
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite training loss at {step}")
            (loss*weight).backward()
            utility_metric += float(uloss.detach())*weight
            action_metric += float(aloss.detach())*weight
        norm = torch.nn.utils.clip_grad_norm_(params.values(),1.)
        if not torch.isfinite(norm):
            raise FloatingPointError(f"Nonfinite gradient at {step}")
        optimizer.step()
        with torch.no_grad():
            for name,value in params.items():
                ema[name].lerp_(value,.01)
        step += 1
        if step%config["log_every"]==0 or step==1:
            torch.cuda.synchronize()
            elapsed = time.monotonic()-window_start
            append_event(output / "events.jsonl","train",step=step,total=total,phase=phase,
                         utility_loss=utility_metric,action_loss=action_metric,grad_norm=float(norm),
                         samples_per_second=(step-window_step)*config["batch_size"]/elapsed,
                         peak_vram_bytes=torch.cuda.max_memory_allocated(),
                         learning_rates={g["role"]:g["lr"] for g in optimizer.param_groups})
            window_start,window_step = time.monotonic(),step
        if step==1 or step%config["save_every"]==0 or step==total or stop_requested():
            save(periodic=(step==1 or step%config["periodic_every"]==0 or step==total))
        if phase=="B" and local_step==99:
            atomic_json(output / "B_100_updates.json",{"step":step,"peak_vram_bytes":torch.cuda.max_memory_allocated(),
                        "batch_size":config["batch_size"],"micro_batch_size":micro,"finite":True})
        if stop_requested():
            return
    atomic_json(output / "COMPLETE.json",{"identity":identity,"step":step,"checkpoint":str(latest),
                                          "label_identity":label_manifest["identity"]})
    append_event(output / "events.jsonl","train_complete",step=step,checkpoint=str(latest))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",required=True)
    parser.add_argument("--output",required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    output = Path(args.output)
    output.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(config["seed"])
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    if (output / "COMPLETE.json").exists():
        return
    preflight = output / "preflight"
    if not (preflight / "COMPLETE.json").exists():
        from .preflight import ftp1
        preflight.mkdir(exist_ok=True)
        ftp1(config,preflight)
        gc.collect()
        torch.cuda.empty_cache()
    if stop_requested():
        return
    data = FTPData(config["task"],config["cache"],config["teacher"])
    config["teacher_sha256"] = file_hash(Path(config["teacher"]) / "model.safetensors")
    model,_,_ = load_released(config["teacher"],rank=config["lora_rank"])
    label_directory = generate_labels(model,data,config,output)
    if label_directory is None or stop_requested():
        return
    model.presence,model.uvip_enabled = None,True
    # Labels only used the untouched release + zero new LoRA; training starts
    # from exactly those inherited parameters and identity FiLM.
    train(model,data,config,output,label_directory)


if __name__=="__main__":
    main()
