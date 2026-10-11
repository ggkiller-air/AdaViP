#!/usr/bin/env python3
"""Compare saved Power Plug action probes without running policy inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_arrays(path: Path) -> dict[str, np.ndarray]:
    """Read a trusted numeric probe archive without pickle deserialization."""
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def compare_probes(root: Path, output_dir: Path) -> dict[str, object]:
    """Check paired inputs and plot action sensitivity on shared color scales."""
    if output_dir.exists():
        raise FileExistsError(f"Output already exists: {output_dir}")
    names = ("baseline", "hyperresnet")
    actions = {name: load_arrays(root / "gpu5_action_probes" /
               f"{name}_epoch100_demos012" / "action_sensitivity.npz") for name in names}
    spatial = {name: load_arrays(root / "gpu5_action_occlusion" /
               f"{name}_epoch100_episode1" / "action_occlusion.npz") for name in names}
    for key in ("episode", "episode_step", "observation_indices", "delayed_observation_indices",
                "seeds", "action_scale"):
        if not np.array_equal(actions[names[0]][key], actions[names[1]][key]):
            raise ValueError(f"action probes are not paired: {key}")
    for key in ("inputs", "steps", "seeds", "modalities"):
        if not np.array_equal(spatial[names[0]][key], spatial[names[1]][key]):
            raise ValueError(f"spatial probes are not paired: {key}")
    output_dir.mkdir(parents=True)
    summary = {name: {str(variant): float(data["rms_all"][:, :, index].mean())
                     for index, variant in enumerate(data["variants"])} for name, data in actions.items()}
    shared_variants = [variant for variant in summary[names[0]] if variant != "repeat"]
    modulation_variants = [variant for variant in summary[names[1]] if variant not in summary[names[0]]]
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
    positions = np.arange(len(shared_variants))
    for model_index, name in enumerate(names):
        bars = axes[0].bar(positions + (model_index - 0.5) * 0.36,
                          [summary[name][variant] for variant in shared_variants], width=0.36, label=name)
        axes[0].bar_label(bars, fmt="%.3f", fontsize=8, padding=3)
    axes[0].set_xticks(positions, shared_variants, rotation=25, ha="right")
    axes[0].legend()
    axes[0].set_title("Same demo histories and paired diffusion noise")
    bars = axes[1].bar(modulation_variants, [summary[names[1]][variant] for variant in modulation_variants])
    axes[1].bar_label(bars, fmt="%.4f", padding=3)
    axes[1].tick_params(axis="x", labelrotation=20)
    axes[1].set_title("HyperResNet modulation-only changes")
    for axis in axes:
        axis.set_ylabel("Normalized action RMS change (not utility)")
        axis.margins(y=0.18)
    figure.savefig(output_dir / "action_sensitivity_comparison.png", dpi=170)
    plt.close(figure)
    reference = spatial[names[0]]
    maps = {name: data["rms_all"].mean(axis=0) for name, data in spatial.items()}
    maximum = max(float(values.max()) for values in maps.values())
    steps = reference["steps"]
    figure, axes = plt.subplots(6, len(steps), figsize=(4 * len(steps), 17), squeeze=False,
                               layout="constrained")
    image = None
    for step_index, step in enumerate(steps):
        for modality_index in range(len(reference["modalities"])):
            rgb = reference["inputs"][step_index, modality_index].transpose(1, 2, 0)
            row = modality_index * 3
            axes[row, step_index].imshow(rgb)
            label = "Wrist vision" if modality_index == 0 else "TacRGB"
            axes[row, step_index].set_title(f"{label}; demo frame {step}")
            for model_index, name in enumerate(names):
                axis = axes[row + model_index + 1, step_index]
                axis.imshow(rgb)
                image = axis.imshow(maps[name][step_index, modality_index], cmap="magma", alpha=0.65,
                    vmin=0, vmax=maximum, interpolation="nearest",
                    extent=(-0.5, rgb.shape[1] - 0.5, rgb.shape[0] - 0.5, -0.5))
                axis.set_title(name)
    for axis in axes.flat:
        axis.set_axis_off()
    figure.colorbar(image, ax=axes, shrink=0.6, label="Normalized action RMS; one shared scale")
    figure.suptitle("Local patch blur: action sensitivity, not attention or success attribution")
    figure.savefig(output_dir / "action_occlusion_comparison.png", dpi=170)
    plt.close(figure)
    report = {"action_means": summary, "occlusion_color_max": maximum,
              "input_pairing": "Exact matching histories, normalizer scales, image inputs and sampling seeds",
              "scope": "Exploratory demo action sensitivity; not causal task utility or independent trials."}
    (output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    """Parse result paths and compare completed probes on CPU."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    compare_probes(args.results_root, args.output_dir)


if __name__ == "__main__":
    main()
