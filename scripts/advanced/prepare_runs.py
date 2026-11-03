"""Freeze seven native-model UViP runs without touching the three DP routes."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root",required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    root = Path(args.root)
    root.mkdir(parents=True,exist_ok=False)
    source = root / "source"
    source.mkdir()
    ignore = shutil.ignore_patterns("__pycache__","*.pyc")
    shutil.copytree(repo / "adavip",source / "adavip",ignore=ignore)
    shutil.copytree(repo / "slurm/advanced",source / "slurm/advanced",ignore=ignore)
    sources = {"ftp1":Path("/data/wangzihao/outputs/uvip_final_implementation_20261102/source/ftp1-policy-main"),
               "tabero":Path("/data/wangzihao/outputs/tabero/preparation_20261102/source/Tabero-VTLA-main")}
    for model,native in sources.items():
        shutil.copytree(native / "src",source / f"native/{model}/src",ignore=ignore)
        shutil.copytree(native / "packages/openpi-client/src",source / f"native/{model}/packages/openpi-client/src",ignore=ignore)
        for name in ("LICENSE","LICENSE_GEMMA.txt"):
            shutil.copy2(native / name,source / f"native/{model}/{name}")
    hashes = []
    for path in sorted(source.rglob("*")):
        if path.is_file():
            hashes.append(hashlib.sha256(path.read_bytes()).hexdigest()+"  "+str(path.relative_to(source)))
    (source / "source.sha256").write_text("\n".join(hashes)+"\n")
    identity = hashlib.sha256((source / "source.sha256").read_bytes()).hexdigest()
    tasks = ["insert_hole","insert_tube","lift_bottle","lift_can","pull_out_key","put_bottle_in_shelf"]
    runs = []
    for task in tasks:
        published = "put_bottle" if task=="put_bottle_in_shelf" else task
        output = root / "ftp1" / task
        output.mkdir(parents=True)
        config = {"task":task,"teacher":f"/data/wangzihao/checkpoints/ftp1/univtac_finetune/FTP1_UniVTAC_{published}_expert_gsmall_ftp1/19999",
                  "cache":f"/data/wangzihao/datasets/univtac/cache/ftp1_native_bgr224_mix_v1/{task}",
                  "source_id":identity,"seed":42,"lora_rank":16,"lora_alpha":16,"utility_k":8,
                  "score_batch_size":8,"batch_size":64,"micro_batch_size":8,
                  "phase_a_steps":1000,"phase_b_steps":9000,"phase_c_steps":10000,
                  "learning_rates":{"utility":1e-4,"uvip":1e-4,"action_lora":5e-5},
                  "utility_loss_weight":.05,"log_every":20,"save_every":100,"periodic_every":1000,
                  "ema_decay":.99,"archive":f"/data/wangzihao/archives/advanced_uvip/{root.name}/ftp1/{task}"}
        path = output / "launch_config.json"
        path.write_text(json.dumps(config,indent=2)+"\n")
        runs.append({"model":"ftp1","task":task,"output":str(output),"config":str(path)})
    output = root / "tabero"
    output.mkdir()
    checkpoint = Path("/data/wangzihao/checkpoints/tabero/pi0_lora_tacfield_tabero/checkpoints/pi0_lora_tacfield_tabero/pi0_lora_tacfield_tabero/49999")
    config = {"teacher":str(checkpoint / "params"),"normalizer":str(checkpoint / "assets/NathanWu7/tabero/norm_stats.json"),
              "teacher_revision":"1ce3438ce5f290da65aff286935ecf97e9ea4046",
              "cache":"/data/wangzihao/datasets/tabero/cache/field_fs_rgb_u8_v1",
              "source_id":identity,"seed":42,"lora_rank":32,"utility_k":8,
              "score_batch_size":8,"batch_size":32,"micro_batch_size":4,
              "phase_a_steps":2000,"phase_b_steps":28000,"phase_c_steps":20000,
              "learning_rates":{"utility":1e-4,"uvip":1e-4,"action_lora":2.5e-5},
              "utility_loss_weight":.05,"log_every":20,"save_every":100,"periodic_every":2500,
              "ema_decay":None,"archive":f"/data/wangzihao/archives/advanced_uvip/{root.name}/tabero"}
    path = output / "launch_config.json"
    path.write_text(json.dumps(config,indent=2)+"\n")
    runs.append({"model":"tabero","output":str(output),"config":str(path)})
    registry = {"source":str(source),"source_id":identity,"created":time.strftime("%Y-%m-%d %H:%M:%S %z"),
                "progress_document":"/public/home/wangzihao/projects/UVIP_5_Routes_Progress_2026-11-03.md","runs":runs}
    (root / "registry.json").write_text(json.dumps(registry,indent=2)+"\n")
    print(root / "registry.json")


if __name__=="__main__":
    main()
