#!/usr/bin/env python3
"""Generate frozen language embeddings for ManiFeel task names."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np

from adavip.manifeel.task_protocol import DEFAULT_TASK_SPECS


def hash_text_embedding(text: str, dim: int) -> np.ndarray:
    """Return a deterministic debug embedding, not a language-model embedding."""
    vector = np.zeros(dim, dtype=np.float32)
    normalized = f" {text.lower().strip()} "
    grams = [normalized[i : i + 3] for i in range(max(1, len(normalized) - 2))]
    for gram in grams:
        digest = hashlib.sha256(gram.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "little") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[index] += sign
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector /= norm
    return vector


def transformer_text_embeddings(
    texts: list[str],
    model_name: str,
    device: str,
    local_files_only: bool = False,
) -> np.ndarray:
    """Encode task descriptions with a frozen Hugging Face text encoder."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
    model = AutoModel.from_pretrained(model_name, local_files_only=local_files_only)
    model.to(device)
    model.eval()

    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.no_grad():
        if hasattr(model, "get_text_features"):
            pooled = model.get_text_features(**encoded)
        else:
            outputs = model(**encoded)
            hidden = outputs.last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=-1)
    return pooled.cpu().numpy().astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, help="Output .npz path.")
    parser.add_argument(
        "--backend",
        choices=("transformer", "hash"),
        default="transformer",
        help="Embedding backend. Use hash only for offline smoke tests.",
    )
    parser.add_argument(
        "--model",
        default="openai/clip-vit-base-patch32",
        help="Frozen Hugging Face encoder used when --backend=transformer.",
    )
    parser.add_argument("--device", default="cpu", help="Torch device for transformer encoding.")
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Require the Hugging Face model to already exist in the local cache.",
    )
    parser.add_argument("--dim", type=int, default=512, help="Hash embedding dimension.")
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    task_ids = np.array([spec.task_id for spec in DEFAULT_TASK_SPECS])
    task_texts = np.array([spec.text for spec in DEFAULT_TASK_SPECS])
    if args.backend == "transformer":
        embeddings = transformer_text_embeddings(
            task_texts.tolist(),
            model_name=args.model,
            device=args.device,
            local_files_only=args.local_files_only,
        )
        model_name = args.model
    else:
        embeddings = np.stack([hash_text_embedding(spec.text, args.dim) for spec in DEFAULT_TASK_SPECS])
        model_name = f"hash-ngram-{args.dim}"

    np.savez(
        output,
        task_ids=task_ids,
        task_texts=task_texts,
        embeddings=embeddings.astype(np.float32),
        backend=np.array(args.backend),
        model=np.array(model_name),
    )
    print(
        f"Wrote {len(task_ids)} {args.backend} task embeddings "
        f"with dim={embeddings.shape[1]} to {output}"
    )


if __name__ == "__main__":
    main()
