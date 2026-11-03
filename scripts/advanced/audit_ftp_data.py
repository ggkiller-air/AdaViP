"""Compare cache/action protocol with executable pinned upstream functions.

Extract only pure functions with AST so the native parser's unrelated zarr/video
dependencies do not influence this read-only CPU data correctness audit.
"""

from __future__ import annotations

import ast
import json
import logging
from pathlib import Path
import types
from typing import Optional

import cv2
import h5py
import numpy as np


def main():
    source = Path("/data/wangzihao/outputs/uvip_final_implementation_20261102/source/ftp1-policy-main")
    namespace = {"np":np,"cv2":cv2,"h5py":h5py,"Path":Path,"Optional":Optional,"IMAGE_SIZE":224,
                 "DEFAULT_GRIPPER_IDX":28,"TACTILE_SENSOR_NAME":"GelSightMini","logging":logging,
                 "FTP1_SINGLE_ARM_JOINT_DIM":7,"FTP1_GRIPPER_HAND_SLOT_INDEX":28,
                 "_RAW_JOINT_GLITCH_THRESH":1e6}
    parser = ast.parse((source / "data_processing/parse_data_module/parse_data_univtac.py").read_text())
    funcs = [node for node in parser.body if isinstance(node,ast.FunctionDef) and node.name in {
        "_stream_to_img","_load_univtac_episode"}]
    dataset = ast.parse((source / "src/openpi/dataset_zarr.py").read_text())
    funcs += [node for node in dataset.body if isinstance(node,ast.FunctionDef) and (
        node.name.startswith("_action_joint_rep_") or node.name in {"_canonicalize_action_joint_rep","_replace_joint_glitches"})]
    native_class = next(node for node in dataset.body if isinstance(node,ast.ClassDef) and node.name=="ZarrDataset")
    funcs += [node for node in native_class.body if isinstance(node,ast.FunctionDef) and node.name=="_get_single_arm_data"]
    # Python3.8 needs future annotations for upstream's modern type syntax.
    import __future__
    exec(compile(ast.Module(body=funcs,type_ignores=[]),"pinned_upstream_functions","exec",
                 flags=__future__.annotations.compiler_flag),namespace)
    base = Path("/data/wangzihao/datasets/univtac/cache/ftp1_native_bgr224_mix_v1")
    report = []
    for directory in sorted(base.iterdir()):
        manifest = json.loads((directory / "manifest.json").read_text())
        episode = manifest["episodes"][0]
        native = namespace["_load_univtac_episode"](episode["path"])
        n = episode["end"]-episode["start"]
        images = {name:np.load(directory / f"{name}.npy",mmap_mode="r") for name in (
            "head","wrist","tactile_left","tactile_right")}
        for name,key in (("head","camera_ego_rgb"),("wrist","right_wrist_camera_rgb")):
            np.testing.assert_array_equal(images[name][:n],native[key])
        np.testing.assert_array_equal(images["tactile_left"][:n],native["right_tactile_data_gripper"][:,0])
        np.testing.assert_array_equal(images["tactile_right"][:n],native["right_tactile_data_gripper"][:,1])
        joints = np.load(directory / "joint.npy",mmap_mode="r")
        stub = types.SimpleNamespace(disable_history=True,state_hisory_length=1,tactile_hisory_length=1,
                                     single_arm_action_rep_dim=48,single_hand_num_tactile_areas=24,
                                     proprioception_joint_rep="abs",action_joint_rep="mix",
                                     proprioception_pose_rep="relative",action_pose_rep="relative")
        for anchor in (0,n//2,n-1):
            target_ids = np.arange(anchor,min(anchor+32,n))
            values = namespace["_get_single_arm_data"](
                stub,native,list(native),set(native),"right",np.array([anchor]),np.array([anchor]),
                target_ids,0,n,np.eye(4),compute_contact=False)
            state,actions,*_,mask = values
            expected_state = np.zeros((1,48),np.float64)
            expected_state[0,9:16],expected_state[0,44] = joints[anchor,:7],joints[anchor,7]
            expected_action = np.zeros((len(target_ids),48),np.float64)
            expected_action[:,9:16] = joints[target_ids,:7].astype(np.float64)-joints[anchor,:7].astype(np.float64)
            expected_action[:,44] = joints[target_ids,7]
            expected_mask = np.zeros_like(expected_action)
            expected_mask[:,9:16],expected_mask[:,44] = 1,1
            np.testing.assert_array_equal(state,expected_state)
            np.testing.assert_array_equal(actions,expected_action)
            np.testing.assert_array_equal(mask,expected_mask)
        report.append({"task":manifest["task"],"native_parser_pixels":"exact",
                       "native_state_action_mask_three_anchors":"exact","frames":manifest["total_steps"]})
        print(report[-1],flush=True)
    Path("/data/wangzihao/outputs/uvip_final_implementation_20261102/ftp1_native_protocol_test.json").write_text(
        json.dumps(report,indent=2)+"\n")


if __name__=="__main__":
    main()
