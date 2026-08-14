"""Task metadata for the ManiFeel 9-task protocol."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class ManiFeelTaskSpec:
    """Static metadata needed to bind a dataset to a simulator task."""

    task_id: str
    text: str
    dataset: str
    isaacgym_cfg_name: str
    action_dim: int


DEFAULT_TASK_SPECS: tuple[ManiFeelTaskSpec, ...] = (
    ManiFeelTaskSpec(
        task_id="peg_insertion",
        text="peg insertion",
        dataset="pih_quan_June06",
        isaacgym_cfg_name="isaacgym_config.yaml",
        action_dim=6,
    ),
    ManiFeelTaskSpec(
        task_id="usb_insertion",
        text="USB insertion",
        dataset="usb_quan_Aug05",
        isaacgym_cfg_name="isaacgym_config_usb.yaml",
        action_dim=6,
    ),
    ManiFeelTaskSpec(
        task_id="power_plug_insertion",
        text="power plug insertion",
        dataset="plug_quan_Aug02",
        isaacgym_cfg_name="isaacgym_config_power_plug.yaml",
        action_dim=6,
    ),
    ManiFeelTaskSpec(
        task_id="gear_assembly",
        text="gear assembly",
        dataset="gear_quan_Sep15",
        isaacgym_cfg_name="isaacgym_config_gear.yaml",
        action_dim=6,
    ),
    ManiFeelTaskSpec(
        task_id="nut_bolt_assembly",
        text="nut and bolt assembly",
        dataset="nutbolt_quan_July1",
        isaacgym_cfg_name="isaacgym_config_nut.yaml",
        action_dim=7,
    ),
    ManiFeelTaskSpec(
        task_id="bulb_installation",
        text="bulb installation",
        dataset="bulb_quan_Sep19",
        isaacgym_cfg_name="isaacgym_config_bulb.yaml",
        action_dim=7,
    ),
    ManiFeelTaskSpec(
        task_id="peg_reorientation",
        text="peg reorientation",
        dataset="blindinsert_quan_Aug15",
        isaacgym_cfg_name="isaacgym_config_peg_reorientation.yaml",
        action_dim=6,
    ),
    ManiFeelTaskSpec(
        task_id="object_search",
        text="object search",
        dataset="explore_quan_June17",
        isaacgym_cfg_name="isaacgym_config_object_search.yaml",
        action_dim=7,
    ),
    ManiFeelTaskSpec(
        task_id="ball_sorting",
        text="ball sorting",
        dataset="sorting_quan_Aug8",
        isaacgym_cfg_name="isaacgym_config_ball_sorting.yaml",
        action_dim=7,
    ),
)


def coerce_task_specs(task_specs: Iterable[dict] | None) -> list[ManiFeelTaskSpec]:
    """Convert Hydra task dictionaries into typed task specs."""
    if task_specs is None:
        return list(DEFAULT_TASK_SPECS)
    specs: list[ManiFeelTaskSpec] = []
    for item in task_specs:
        specs.append(
            ManiFeelTaskSpec(
                task_id=str(item["task_id"]),
                text=str(item["text"]),
                dataset=str(item["dataset"]),
                isaacgym_cfg_name=str(item["isaacgym_cfg_name"]),
                action_dim=int(item["action_dim"]),
            )
        )
    return specs
