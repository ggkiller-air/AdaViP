"""Update the user's five-route Markdown from committed run artifacts."""

from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
import re


def mark_ready_stages(text):
    """Mark the next unstarted stage human once its predecessor is complete."""
    routes = {"ManiFeel DP + UViP", "UniVTAC DP + UViP", "Tabero DP + UViP",
              "UniVTAC FTP-1 + UViP", "Tabero-VTLA + UViP"}
    lines = text.splitlines()
    for index, line in enumerate(lines):
        cells = [cell.strip() for cell in line.split("|")]
        if len(cells) != 8 or cells[1] not in routes:
            continue
        for column in range(3, 7):
            if cells[column] == "pending" and cells[column - 1] == "√completed":
                cells[column] = "human"
        lines[index] = "| " + " | ".join(cells[1:-1]) + " |"
    return "\n".join(lines) + "\n"


def update():
    registry_path = os.environ.get("UVIP_RUN_REGISTRY")
    if not registry_path:
        return
    registry_path = Path(registry_path)
    registry = json.loads(registry_path.read_text())
    document = Path(registry["progress_document"])
    states = {}
    for line in subprocess.check_output(["squeue","-u","wangzihao","-h","-o","%i %T"],text=True).splitlines():
        parts = line.split()
        if len(parts)==2:
            states[parts[0]] = parts[1]
    with open(registry_path.with_suffix(".progress.lock"),"w") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        text = document.read_text()
        text = re.sub(r"更新时间：[^\n]+",f"更新时间：{time.strftime('%Y-%m-%d %H:%M:%S %z')}。以服务器文件、Slurm和验证报告为准。",text,count=1)
        dp_paths = {
            "UniVTAC DP + UViP":"/data/wangzihao/experiments/univtac/l1_six_task_batch64_ep500_seed42_20261102_233553/train/COMPLETE.json",
            "Tabero DP + UViP":"/data/wangzihao/experiments/tabero/l1_field_fs_batch64_ep200_seed42_20261102_234707/train/COMPLETE.json"}
        lines = text.splitlines()
        for index,line in enumerate(lines):
            for route,path in dp_paths.items():
                if line.startswith(f"| {route} |") and Path(path).exists():
                    cells = [c.strip() for c in line.split("|")]
                    cells[2] = "√completed"
                    lines[index] = "| "+" | ".join(cells[1:-1])+" |"
        text = "\n".join(lines)+"\n"
        for run in registry.get("dp_stage2_runs", []):
            root = Path(run["output"])
            status = ("√completed" if (root / "COMPLETE.json").exists() else
                      "running" if (root / "contract.json").exists() and
                      states.get(str(run.get("job_id"))) in {"RUNNING", "COMPLETING"} else "pending")
            lines = text.splitlines()
            for index, line in enumerate(lines):
                if line.startswith(f"| {run['route']} |"):
                    cells = [cell.strip() for cell in line.split("|")]
                    cells[5] = status
                    lines[index] = "| " + " | ".join(cells[1:-1]) + " |"
            text = "\n".join(lines) + "\n"
        rows = []
        by_model = {"ftp1":[],"tabero":[]}
        for run in registry["runs"]:
            root = Path(run["output"])
            labels_done = (root / "labels/COMPLETE.json").exists()
            labels_started = (root / "labels/contract.json").exists()
            training_done = (root / "COMPLETE.json").exists()
            training_started = (root / "training_contract.json").exists()
            failed = (root / "FAILED.json").exists()
            executing = states.get(str(run.get("job_id"))) in {"RUNNING","COMPLETING"}
            record = {"labels_done":labels_done,"labels_started":labels_started,
                      "training_done":training_done,"training_started":training_started,"failed":failed,"executing":executing}
            by_model[run["model"]].append(record)
            preflight = "√completed" if (root / "preflight/COMPLETE.json").exists() else "pending"
            label_state = "√completed" if labels_done else "running" if labels_started and not failed and executing else "pending"
            train_state = "√completed" if training_done else "running" if training_started and not failed and executing else "pending"
            rows.append(f"| {run['model']} / {run.get('task','multitask')} | {run.get('job_id','pending')} | {preflight} | {label_state} | {train_state} | `{root}` |")
        for model,route in (("ftp1","UniVTAC FTP-1 + UViP"),("tabero","Tabero-VTLA + UViP")):
            entries = by_model[model]
            if not entries:
                continue
            label_state = "√completed" if all(e["labels_done"] for e in entries) else "running" if any(
                e["labels_started"] and not e["failed"] and not e["labels_done"] and e["executing"] for e in entries) else "pending"
            train_state = "√completed" if all(e["training_done"] for e in entries) else "running" if any(
                e["training_started"] and not e["failed"] and not e["training_done"] and e["executing"] for e in entries) else "pending"
            lines = text.splitlines()
            for index,line in enumerate(lines):
                if line.startswith(f"| {route} |"):
                    cells = [c.strip() for c in line.split("|")]
                    cells[4],cells[5] = label_state,train_state
                    lines[index] = "| "+" | ".join(cells[1:-1])+" |"
            text = "\n".join(lines)+"\n"
        block = ("<!-- ADVANCED_RUNS_BEGIN -->\n## 高级方法运行记录（自动更新）\n\n"
                 "| 路线 / 任务 | Job ID | 全模型验证 | 标签 | Stage 2 | 输出目录 |\n"
                 "|---|---|---|---|---|---|\n"+"\n".join(rows)+
                 f"\n\n最后同步：{time.strftime('%Y-%m-%d %H:%M:%S %z')}。失败原因见各目录 `FAILED.json`；标签/训练详情见 `events.jsonl`。\n"
                 "<!-- ADVANCED_RUNS_END -->")
        if "<!-- ADVANCED_RUNS_BEGIN -->" in text:
            first = text.index("<!-- ADVANCED_RUNS_BEGIN -->")
            last = text.index("<!-- ADVANCED_RUNS_END -->")+len("<!-- ADVANCED_RUNS_END -->")
            text = text[:first]+block+text[last:]
        else:
            text += "\n"+block+"\n"
        text = mark_ready_stages(text)
        temporary = document.with_name(document.name+f".{os.getpid()}.tmp")
        temporary.write_text(text)
        os.replace(temporary,document)


if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch",action="store_true")
    parser.add_argument("--interval",type=int,default=300)
    args = parser.parse_args()
    while True:
        update()
        if not args.watch:
            break
        registry = json.loads(Path(os.environ["UVIP_RUN_REGISTRY"]).read_text())
        if all((Path(run["output"]) / "COMPLETE.json").exists() or
               (Path(run["output"]) / "FAILED.json").exists() for run in registry["runs"]):
            break
        time.sleep(args.interval)
