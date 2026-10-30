"""Compare convolution runtime profiles with batch8 and fixed scientific settings."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import signal
import time

import dill
import hydra
import torch

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.model.diffusion.ema_model import EMAModel


def main() -> None:
    """Pause only the specified trainer, benchmark its allocation, then resume it."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--trainer-pid", type=int, required=True)
    parser.add_argument("--training-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=30)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    command = Path(f"/proc/{args.trainer_pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
    if "train_single_task_teacher.py" not in command or str(args.training_output) not in command:
        raise RuntimeError("Specified PID is not this experiment's trainer")

    def interrupted(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    os.kill(args.trainer_pid, signal.SIGSTOP)
    paused = time.time()
    try:
        time.sleep(2)
        payload = torch.load(args.checkpoint, map_location="cpu", pickle_module=dill)
        cfg = payload["cfg"]
        state = payload["state_dicts"]["model"]
        del payload
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        batch = torch.utils.data.default_collate([dataset[index] for index in range(8)])
        batch = dict_apply(batch, lambda value: value.to("cuda:0"))
        torch.set_num_threads(4)
        records = []
        flags = {"matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
                 "cudnn_tf32": torch.backends.cudnn.allow_tf32,
                 "trainer_cpus": sorted(os.sched_getaffinity(0)), "batch_size": 8}
        for name, autotune, channels_last in (("baseline", False, False),
                                              ("cudnn_benchmark", True, False),
                                              ("channels_last", True, True)):
            torch.backends.cudnn.benchmark = autotune
            model = hydra.utils.instantiate(cfg.policy)
            model.load_state_dict(state, strict=True)
            model.to("cuda:0").train()
            if channels_last:
                for backbone in model.obs_encoder.image_encoders.values():
                    backbone.to(memory_format=torch.channels_last)
                    backbone.register_forward_pre_hook(lambda module, inputs: (
                        inputs[0].contiguous(memory_format=torch.channels_last),))
            ema_policy = copy.deepcopy(model).requires_grad_(False).eval()
            ema = EMAModel(ema_policy, power=0.75)
            optimizer = hydra.utils.instantiate(cfg.optimizer, params=model.parameters())
            torch.manual_seed(42)
            ids = torch.zeros(8, dtype=torch.long, device="cuda:0")
            first_loss = None

            def step():
                nonlocal first_loss
                optimizer.zero_grad(set_to_none=True)
                loss = model.losses(batch, ids).mean()
                if first_loss is None:
                    first_loss = float(loss.detach())
                loss.backward()
                if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
                    raise FloatingPointError("Nonfinite benchmark gradients")
                optimizer.step()
                ema.step(model)

            for _ in range(10):
                step()
            timings = []
            for _ in range(3):
                torch.cuda.synchronize()
                started = time.perf_counter()
                for _ in range(args.steps):
                    step()
                torch.cuda.synchronize()
                timings.append(8 * args.steps / (time.perf_counter() - started))
            record = {"profile": name, "cudnn_benchmark": autotune, "channels_last": channels_last,
                      "samples_per_second": timings, "first_loss": first_loss, **flags}
            records.append(record)
            print(json.dumps(record), flush=True)
            (args.output / "results.json").write_text(json.dumps(records, indent=2) + "\n")
            del optimizer, ema, ema_policy, model
            import gc
            gc.collect()
            torch.cuda.empty_cache()
    finally:
        os.kill(args.trainer_pid, signal.SIGCONT)
        (args.output / "trainer_resumed.json").write_text(json.dumps({"pid": args.trainer_pid,
            "paused_seconds": time.time() - paused, "timestamp": time.time()}) + "\n")


if __name__ == "__main__":
    main()
