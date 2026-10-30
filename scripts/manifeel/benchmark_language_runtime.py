"""Compare LC runtime settings using full data and the production training loop."""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import random
import time

import hydra
from omegaconf import OmegaConf
import torch

from adavip.manifeel.language_conditioning_workspace import LanguageConditioningWorkspace
from scripts.manifeel.language_runtime import configure_profile


def atomic_json(path: Path, value: dict) -> None:
    """Persist completed benchmark cases without exposing a partial report."""
    temporary = path.with_suffix(".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def bind_current_threads(cpus: list[int]) -> None:
    """Apply the measured CPU mask to all existing trainer threads."""
    for task in Path("/proc/self/task").iterdir():
        try:
            os.sched_setaffinity(int(task.name), cpus)
        except ProcessLookupError:
            pass


def sibling_pair(allowed: list[int]) -> list[int]:
    """Find two SMT siblings within the allocated CPU mask."""
    cpu = allowed[0]
    value = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list").read_text().strip()
    siblings = set()
    for group in value.split(","):
        ends = list(map(int, group.split("-")))
        siblings.update(range(ends[0], ends[-1] + 1))
    return sorted(siblings.intersection(allowed))


class MeasuredWorkspace(LanguageConditioningWorkspace):
    """Instrument the unchanged production update and keep test saves isolated."""

    def __init__(self, cfg: object, dataset: object, warmup: int, steps: int,
                 save_checkpoint: bool = False) -> None:
        super().__init__(cfg, output_dir=cfg.output_dir)
        self.benchmark_dataset = dataset
        self.warmup, self.steps = warmup, steps
        self.updates, self.measured_samples = 0, 0
        self.losses: list[float] = []
        self.started, self.seconds = None, None
        self.save_enabled = save_checkpoint
        self.worker_affinities: dict = {}

    def _make_dataset(self) -> object:
        return self.benchmark_dataset

    def _save_latest(self) -> None:
        if self.save_enabled:
            super()._save_latest()

    def _accumulate(self, state: dict, batch: dict, losses: torch.Tensor,
                    condition_ids: torch.Tensor) -> tuple[int, float]:
        result = super()._accumulate(state, batch, losses, condition_ids)
        self.updates += 1
        if self.updates > self.warmup:
            self.measured_samples += result[0]
            self.losses.append(result[1])
        if self.updates == self.warmup:
            iterator = self._persistent_loader._iterator
            self.worker_affinities = {str(w.pid): sorted(os.sched_getaffinity(w.pid))
                                      for w in iterator._workers}
            torch.cuda.synchronize()
            self.started = time.perf_counter()
        if self.updates == self.warmup + self.steps:
            torch.cuda.synchronize()
            self.seconds = time.perf_counter() - self.started
        return result

    def close_workers(self) -> None:
        """Join this case's persistent workers before changing process affinity."""
        if self._persistent_loader is not None and self._persistent_loader._iterator is not None:
            self._persistent_loader._iterator._shutdown_workers()
        self._persistent_loader = None


def run_case(base: object, dataset: object, output: Path, checkpoint: Path,
             name: str, profile: str, cpus: list[int], warmup: int, steps: int,
             save_checkpoint: bool = False) -> dict:
    """Resume identical initial weights, optimizer and random streams for one case."""
    bind_current_threads(cpus)
    flags = configure_profile(profile)
    cfg = OmegaConf.create(OmegaConf.to_container(base, resolve=True))
    cfg.output_dir, cfg.resume = str(output / name), str(checkpoint)
    cfg.runtime.stop_after_steps = warmup + steps
    workspace = MeasuredWorkspace(cfg, dataset, warmup, steps, save_checkpoint)
    try:
        torch.cuda.reset_peak_memory_stats()
        workspace.run()
        if workspace.seconds is None or workspace.updates != warmup + steps:
            raise RuntimeError("Incomplete timed run")
        norms = []
        for line in (Path(cfg.output_dir) / "progress.jsonl").read_text().splitlines():
            row = json.loads(line)
            if row.get("event") == "train_step":
                norms.append(row["grad_norm"])
        return {"name": name, "profile": profile, "flags": flags,
                "cpus": cpus, "worker_affinities": workspace.worker_affinities,
                "steps": steps, "warmup": warmup, "seconds": workspace.seconds,
                "samples_per_second": workspace.measured_samples / workspace.seconds,
                "losses": workspace.losses, "gradient_norms": norms,
                "global_step": workspace.global_step, "batch_in_epoch": workspace.batch_in_epoch,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "checkpoint": str(Path(cfg.output_dir) / "checkpoints/latest.ckpt") if save_checkpoint else None}
    finally:
        workspace.close_workers()
        del workspace
        gc.collect()
        torch.cuda.empty_cache()


def main() -> None:
    """Compare repeated full-cache cases and validate a candidate checkpoint resume."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=24)
    parser.add_argument("--steps", type=int, default=160)
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Requires one independent H200 allocation")
    allowed = sorted(os.sched_getaffinity(0))
    if len(allowed) < 32:
        raise RuntimeError(f"Expected32allocated logicalCPUs, found{allowed}")
    args.output.mkdir(parents=True, exist_ok=True)
    report_path = args.output / "report.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {"cases": []}
    report.update(job_id=os.environ.get("SLURM_JOB_ID"), gpu=torch.cuda.get_device_name(),
                  torch=torch.__version__, allowed_cpus=allowed, batch_size=64,
                  dataset_scope="all71265anchors; verified full299GBRAMcache")
    atomic_json(report_path, report)
    cfg = OmegaConf.load(args.config)
    dataset = hydra.utils.instantiate(cfg.dataset)
    started = time.perf_counter()
    dataset.load_ram_cache()
    report["cache_load_seconds"] = time.perf_counter() - started
    atomic_json(report_path, report)
    print(json.dumps({"event": "full_cache_ready", "seconds": report["cache_load_seconds"]}), flush=True)
    cases = [("baseline", "strict", sibling_pair(allowed)), ("cpu", "strict", allowed),
             ("nondeterministic", "nondeterministic", allowed),
             ("tf32_deterministic", "tf32_deterministic", allowed),
             ("stage0", "stage0", allowed)]
    for repeat in range(2):
        order = list(cases)
        random.Random(42 + repeat).shuffle(order)
        for label, profile, cpus in order:
            name = f"r{repeat}_{label}"
            if any(row["name"] == name for row in report["cases"]):
                continue
            try:
                result = run_case(cfg, dataset, args.output, args.checkpoint, name, profile,
                                  cpus, args.warmup, args.steps)
            finally:
                bind_current_threads(allowed)
            report["cases"].append(result)
            atomic_json(report_path, report)
            print(json.dumps({key: value for key, value in result.items() if key != "losses"}), flush=True)
    import statistics
    grouped = {label: [row for row in report["cases"] if row["name"].endswith("_" + label)]
               for label, _, _ in cases}
    speeds = {label: statistics.median(row["samples_per_second"] for row in rows)
              for label, rows in grouped.items()}
    # Require improvement in both repeats; compare identical noise/data draws.
    candidates = []
    for label, rows in grouped.items():
        if label == "baseline":
            continue
        stable = all(row["samples_per_second"] >= 1.2 * grouped["baseline"][i]["samples_per_second"]
                     for i, row in enumerate(rows))
        loss_ok = all(statistics.mean(row["losses"]) <= 1.25 * statistics.mean(grouped["baseline"][i]["losses"])
                      for i, row in enumerate(rows))
        gradients_ok = all(row["gradient_norms"] and max(row["gradient_norms"]) <=
                           3 * max(grouped["baseline"][i]["gradient_norms"]) + 1e-6
                           for i, row in enumerate(rows))
        if stable and loss_ok and gradients_ok:
            candidates.append(label)
    chosen = max(candidates, key=lambda label: speeds[label]) if candidates else None
    if chosen and "cpu" in candidates and speeds["cpu"] >= 0.95 * speeds[chosen]:
        chosen = "cpu"
    report.update(median_samples_per_second=speeds, selected=chosen, improvement_gate=bool(chosen))
    atomic_json(report_path, report)
    if chosen:
        profile = next(profile for label, profile, _ in cases if label == chosen)
        validation = run_case(cfg, dataset, args.output, args.checkpoint,
                              "checkpoint_validation", profile, allowed, 4, 12, True)
        resumed = run_case(cfg, dataset, args.output, Path(validation["checkpoint"]),
                          "resume_validation", profile, allowed, 1, 3, False)
        report["checkpoint_validation"] = validation
        report["resume_validation"] = resumed
        report["resume_gate"] = resumed["global_step"] == validation["global_step"] + 4
    report["complete"] = True
    atomic_json(report_path, report)
    print(json.dumps({key: value for key, value in report.items() if key not in
                      ("cases", "checkpoint_validation", "resume_validation")}), flush=True)


if __name__ == "__main__":
    main()
