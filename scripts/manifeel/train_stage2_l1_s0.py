"""Train the final ManiFeel L1-E50 -> S0 Student without modifying legacy runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import signal
from pathlib import Path

import cv2
import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

from adavip.manifeel.full_sensor_evaluation import load_evaluation_policy
from adavip.manifeel.full_sensor_workspace import normalizer_fingerprint
from adavip.manifeel.stage2_ablation_workspace import Stage2AblationTrainer, tensor_fingerprint
from adavip.manifeel.stage2_data import make_stage2_dataset, validate_complete_ram_cache, load_complete_ram_cache
from adavip.manifeel.stage2_l1_policy import Stage2L1Policy
from adavip.manifeel.training_runtime import check_dataloader_ipc


TEACHER_SHA256 = "95bd523ecac6986158d2f4a1c09b19c0a58677b00d30c1d0b417f852fdcfa5fb"
LABEL_SHA256 = "775a772d0799f6c6ee6961300ef55b2b51c3277b2ca9fde82f5ab537458bfe79"


def file_sha256(path: Path) -> str:
    """Hash a persistent artifact with bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_teacher(path: Path, labels: Path):
    """Require the published E50 EMA and exact archived label normalizer."""
    if file_sha256(path) != TEACHER_SHA256:
        raise ValueError("Teacher checkpoint SHA256 differs from the published L1 E50")
    teacher, cfg, metadata = load_evaluation_policy(path, weights="ema")
    expected = dict(epoch=50, global_step=55700, batch_in_epoch=0, weights="ema")
    if metadata != expected or teacher.obs_encoder.variant != "l1":
        raise ValueError(f"Wrong Teacher: {metadata}")
    if cfg.policy.full_input_probability != 1.0:
        raise ValueError("Teacher must have no modality dropout")
    state = teacher.normalizer.state_dict()
    with np.load(labels / "metadata/teacher_normalizer.npz", allow_pickle=False) as archived:
        if set(state) != set(archived.files):
            raise ValueError("Teacher/labels normalizer keys differ")
        for key, value in state.items():
            if not np.array_equal(value.cpu().numpy(), archived[key]):
                raise ValueError(f"Teacher/labels normalizer differs: {key}")
    return teacher, cfg


class L1Trainer(Stage2AblationTrainer):
    """Retain first/final snapshots and archive the first recoverable Student."""

    def save(self) -> None:
        super().save()
        latest = self.output / "checkpoints/latest.ckpt"
        if self.global_step == 1:
            archive = Path("/data/wangzihao/archives/manifeel_uvip") / self.output.name
            archive.mkdir(parents=True, exist_ok=True)
            target = archive / "first_update.ckpt"
            if not target.exists():
                import shutil
                temporary = archive / "first_update.ckpt.partial"
                shutil.copyfile(latest, temporary)
                os.replace(temporary, target)
        if self.epoch > self.epochs:
            target = latest.parent / f"epoch{self.epochs}.ckpt"
            if not target.exists():
                os.link(latest, target)


def main() -> None:
    """Validate immutable inputs, then automatically train A -> B -> C."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("teacher", "labels", "cache", "cache-config", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--dataset-root", default="/data/wangzihao/datasets/manifeel")
    parser.add_argument("--task-embeddings", default="/data/wangzihao/outputs/manifeel/task_embeddings/clip_vit_b32_9task.npz")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--cache-workers", type=int, default=8)
    parser.add_argument("--cache-mode", choices=("eager", "streaming"), default="eager")
    parser.add_argument("--resume-data-transport", action="store_true")
    parser.add_argument("--stop-after-steps", type=int, default=0)
    parser.add_argument("--check-inputs", action="store_true")
    parser.add_argument("--ablation", choices=("Full", "B", "C", "D", "E", "F", "C_teacher_residual"), default="Full")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    check_dataloader_ipc()
    cv2.setNumThreads(0)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    dataset = make_stage2_dataset(str(args.labels), args.dataset_root, args.task_embeddings)
    if len(dataset) != 65265 or len(dataset.task_specs) != 9:
        raise ValueError("Expected all 65,265 windows and nine tasks")
    if dataset.store.manifest["sha256"] != LABEL_SHA256:
        raise ValueError("Labels are not the completed L1 E50 K64 labels")
    cache = validate_complete_ram_cache(dataset.source, str(args.cache), str(args.cache_config),
                                        require_full_ram=not args.smoke)
    utility = np.asarray(dataset.store.utility, dtype=np.float64)
    rms = np.maximum(np.sqrt(np.square(utility).mean(axis=0)), 1e-6)
    teacher, cfg = load_teacher(args.teacher, args.labels)
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    cfg.policy._target_ = "adavip.manifeel.stage2_l1_policy.Stage2L1Policy"
    cfg.policy.encoder_strategy = "s0"
    cfg.policy.utility_rms = rms.tolist()
    policy = hydra.utils.instantiate(cfg.policy)
    policy.initialize_from_teacher(teacher)
    if tensor_fingerprint(policy.obs_encoder.state_dict()) != tensor_fingerprint(teacher.obs_encoder.state_dict()):
        raise RuntimeError("Inherited L1 encoder differs from the Teacher")
    args.output.mkdir(parents=True, exist_ok=True)
    ablation_audit, shuffle = None, None
    if args.ablation != "Full":
        from adavip.manifeel.stage2_main_ablation import (
            FULL_INITIALIZATION_SHA256, PERMUTATION_SEED,
            audit_pruned_initialization, save_shuffle_manifest)
        full_state = policy.state_dict()
        canonical = tensor_fingerprint(full_state)
        if canonical != FULL_INITIALIZATION_SHA256:
            raise RuntimeError(f"Cannot reproduce existing Full initialization: {canonical}")
        saved_rng = torch.get_rng_state().clone()
        policy.apply_ablation(args.ablation)
        ablation_audit = audit_pruned_initialization(full_state, policy)
        ablation_audit.update(canonical_full_sha256=canonical, initialization_seed=42,
                              training_seed=42, teacher_weights="ema", teacher_epoch=50,
                              teacher_global_step=55700)
        assert torch.equal(saved_rng, torch.get_rng_state())
        del full_state
        cfg.policy.ablation = args.ablation
        if args.ablation == "C_teacher_residual":
            ablation_audit["common_c_before_teacher_unet_sha256"] = tensor_fingerprint(
                {k: v for k, v in policy.state_dict().items() if not k.startswith("w_out.")})
            if ablation_audit["common_c_before_teacher_unet_sha256"] != "21db0ba749cc21041909ec85f68de14bae63f7693063b72eb03929a22b4fbf03":
                raise RuntimeError("Cannot reproduce original C initialization")
            policy.initialize_from_teacher(teacher)
            ablation_audit["teacher_ema_unet_sha256"] = tensor_fingerprint(teacher.model.state_dict())
            assert ablation_audit["teacher_ema_unet_sha256"] == tensor_fingerprint(policy.model.state_dict())
            ablation_audit["residual_initialization_sha256"] = tensor_fingerprint(policy.state_dict())
            assert torch.equal(saved_rng, torch.get_rng_state())
        if args.ablation == "F":
            shuffle = save_shuffle_manifest(dataset, args.output, PERMUTATION_SEED)
        (args.output / "initialization_audit.json").write_text(json.dumps(ablation_audit, indent=2) + "\n")
    indices = ([int(np.flatnonzero(dataset.task_indices == task)[0]) for task in range(9)]
               if args.smoke and args.ablation == "C_teacher_residual" else [0, len(dataset) - 1])
    batch = torch.utils.data.default_collate([dataset[index] for index in indices])
    with torch.no_grad():
        expected = teacher.obs_encoder(teacher.normalizer.normalize(batch["obs"]))
        actual = policy.encode_stage2(batch["obs"])["z_t"]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    if args.smoke and args.ablation == "C_teacher_residual":
        from adavip.manifeel.teacher_residual_validation import validate_teacher_parity
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        parity = validate_teacher_parity(teacher, policy, batch, device="cuda:0")
        (args.output / "teacher_parity.json").write_text(json.dumps(parity, indent=2) + "\n")
        policy.cpu()
    del teacher
    root = Path(__file__).resolve().parents[2]
    source_digest = hashlib.sha256()
    for directory in ("adavip/manifeel", "scripts/manifeel"):
        for path in sorted((root / directory).glob("*.py")):
            source_digest.update(str(path.relative_to(root)).encode())
            source_digest.update(path.read_bytes())
    contract = dict(
        variant="stage2_transformer", encoder_variant="l1", encoder_strategy="s0",
        teacher=str(args.teacher.resolve()), teacher_sha256=TEACHER_SHA256,
        labels_sha256=LABEL_SHA256, cache_contract=cache["contract"], cache_bytes=cache["nbytes"],
        task_embedding_sha256=dataset.source.manifest["task_embedding_sha256"],
        batch_size=32, epochs=600, windows=65265, seed=42, utility_weight=0.05, rms=rms.tolist(),
        normalizer=normalizer_fingerprint(policy.normalizer), source_sha256=source_digest.hexdigest(),
        encoder_lr=8e-5, utility_lr=1e-4, action_lr=1e-4, encoder_start_epoch=7,
        encoder_ramp_updates=500, warmup_updates=500, gradient_clip_norm=1.0, ema_decay=0.9999,
        checkpoint_epochs=[1, 6, 10, 20, 50, 100, 200, 300, 400, 500, 600])
    if args.ablation != "Full":
        contract.update(ablation=args.ablation, initialization_audit=ablation_audit,
                        shuffle=shuffle, teacher_weights="ema", teacher_epoch=50,
                        teacher_global_step=55700, initialization_seed=42, training_seed=42,
                        training_windows="all original 65265; no held-out validation",
                        skipped_a_steps=2040 if args.ablation in ("B", "E") else 0,
                        stage_b_updates=10200, stage_c_updates=1211760,
                        global_step_semantics="original Full protocol clock; includes skipped A slots")
    if args.ablation == "C_teacher_residual":
        provenance = root / "source_provenance.json"
        if provenance.exists():
            contract["source_provenance"] = json.loads(provenance.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    if args.resume_data_transport:
        if args.resume is None or args.cache_mode != "streaming":
            raise ValueError("Transport migration requires streaming mode and an explicit resume")
        from adavip.manifeel.streaming_cache import preserve_training_contract
        previous = json.loads((args.output / "contract.json").read_text())
        actual_source = contract["source_sha256"]
        contract = preserve_training_contract(contract, previous)
        with (args.output / "data_transport_migrations.jsonl").open("a") as stream:
            stream.write(json.dumps({"mode": "streaming", "runtime_source_sha256": actual_source,
                                     "original_source_sha256": contract["source_sha256"],
                                     "resume": str(args.resume)}) + "\n")
    (args.output / "inputs_verified.json").write_text(json.dumps(contract, indent=2) + "\n")
    print(json.dumps(dict(event="l1_e50_inputs_verified", windows=len(dataset), cache_bytes=cache["nbytes"])), flush=True)
    if args.check_inputs:
        return
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or "H200" not in torch.cuda.get_device_name(0):
        raise RuntimeError("Formal training requires exactly one allocated H200")
    if args.smoke:
        from scripts.manifeel.smoke_stage2_main_ablation import run_smoke
        run_smoke(policy, dataset, args.output, contract, OmegaConf.to_container(cfg, resolve=True), args)
        return
    trainer_class = L1Trainer
    if args.ablation != "Full":
        from adavip.manifeel.stage2_main_ablation import MainAblationTrainer
        trainer_class = MainAblationTrainer
    cache_loader = lambda: load_complete_ram_cache(dataset.source, str(args.cache),
                                                  str(args.cache_config), args.cache_workers)
    if args.cache_mode == "streaming":
        from adavip.manifeel.streaming_cache import StreamingCache
        cache_loader = StreamingCache(dataset.source, str(args.cache), cache, args.output)
    trainer = trainer_class(policy, dataset, args.output, contract, OmegaConf.to_container(cfg, resolve=True),
                            cache_loader, workers=args.workers, stop_after_steps=args.stop_after_steps)
    for signum in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, lambda *_: setattr(trainer, "stop_requested", True))
    try:
        trainer.run(args.resume)
    finally:
        if hasattr(cache_loader, "close"):
            cache_loader.close()


if __name__ == "__main__":
    main()
