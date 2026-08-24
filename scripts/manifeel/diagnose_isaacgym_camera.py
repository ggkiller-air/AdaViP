#!/usr/bin/env python3
"""Isolate Isaac Gym camera rendering from ManiFeel and TacSL."""

import argparse
import time
from typing import Any, List, Tuple

from isaacgym import gymapi, gymtorch


def log(message: str) -> None:
    """Print a timestamped diagnostic marker immediately."""

    print(f"[{time.monotonic():.6f}] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    """Parse the deliberately small camera diagnostic surface."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--access", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--image-type", choices=("color", "depth"), default="color")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--frames", type=int, default=1)
    parser.add_argument("--compute-device-id", type=int, default=0)
    parser.add_argument("--graphics-device-id", type=int, default=0)
    parser.add_argument("--acquire-after-prepare", action="store_true")
    parser.add_argument("--attach-camera", action="store_true")
    return parser.parse_args()


def image_type_from_name(name: str) -> gymapi.ImageType:
    """Map the CLI image type to the Isaac Gym selector."""

    if name == "color":
        return gymapi.IMAGE_COLOR
    return gymapi.IMAGE_DEPTH


def acquire_gpu_tensors(
    gym: Any,
    sim: Any,
    cameras: List[Tuple[Any, int]],
    image_type: gymapi.ImageType,
) -> List[Any]:
    """Acquire and wrap every configured GPU camera tensor."""

    tensors = []
    for index, (env, camera) in enumerate(cameras):
        log(f"gpu_acquire_begin env={index}")
        descriptor = gym.get_camera_image_gpu_tensor(sim, env, camera, image_type)
        log(f"gpu_acquire_end env={index} shape={descriptor.shape}")
        tensors.append(gymtorch.wrap_tensor(descriptor))
        log(f"gpu_wrap_end env={index} shape={tuple(tensors[-1].shape)}")
    return tensors


def main() -> None:
    """Build a minimal scene and exercise exactly one camera access path."""

    args = parse_args()
    if args.num_envs < 1:
        raise ValueError("--num-envs must be positive")
    if args.frames < 1:
        raise ValueError("--frames must be positive")

    image_type = image_type_from_name(args.image_type)
    gym = gymapi.acquire_gym()
    sim_params = gymapi.SimParams()
    sim_params.physx.use_gpu = True
    sim_params.use_gpu_pipeline = True

    log("create_sim_begin")
    sim = gym.create_sim(
        args.compute_device_id,
        args.graphics_device_id,
        gymapi.SIM_PHYSX,
        sim_params,
    )
    if sim is None:
        raise RuntimeError("gym.create_sim returned None")
    log("create_sim_end")

    sphere = gym.create_sphere(sim, 0.05, gymapi.AssetOptions())
    cameras: List[Tuple[Any, int]] = []
    lower = gymapi.Vec3(-1.0, -1.0, -1.0)
    upper = gymapi.Vec3(1.0, 1.0, 1.0)
    per_row = max(1, int(args.num_envs ** 0.5))

    for index in range(args.num_envs):
        env = gym.create_env(sim, lower, upper, per_row)
        pose = gymapi.Transform()
        actor = gym.create_actor(env, sphere, pose, "sphere", index, 0)

        properties = gymapi.CameraProperties()
        properties.width = 128
        properties.height = 128
        properties.enable_tensors = args.access == "gpu"
        camera = gym.create_camera_sensor(env, properties)
        if args.attach_camera:
            body = gym.get_actor_rigid_body_handle(env, actor, 0)
            transform = gymapi.Transform(gymapi.Vec3(0.0, 0.0, 0.2))
            gym.attach_camera_to_body(
                camera, env, body, transform, gymapi.FOLLOW_TRANSFORM
            )
        else:
            gym.set_camera_location(
                camera,
                env,
                gymapi.Vec3(0.3, 0.3, 0.3),
                gymapi.Vec3(0.0, 0.0, 0.0),
            )
        cameras.append((env, camera))
        log(f"camera_created env={index}")

    tensors: List[Any] = []
    if args.access == "gpu" and not args.acquire_after_prepare:
        tensors = acquire_gpu_tensors(gym, sim, cameras, image_type)

    log("prepare_sim_begin")
    gym.prepare_sim(sim)
    log("prepare_sim_end")

    if args.access == "gpu" and args.acquire_after_prepare:
        tensors = acquire_gpu_tensors(gym, sim, cameras, image_type)

    for frame in range(args.frames):
        log(f"frame_begin frame={frame}")
        gym.simulate(sim)
        gym.fetch_results(sim, True)
        gym.step_graphics(sim)
        gym.render_all_camera_sensors(sim)

        if args.access == "gpu":
            gym.start_access_image_tensors(sim)
            try:
                tensors[0].clone()
                log(f"gpu_access_end frame={frame} shape={tuple(tensors[0].shape)}")
            finally:
                gym.end_access_image_tensors(sim)
        else:
            env, camera = cameras[0]
            log(f"cpu_access_begin frame={frame}")
            image = gym.get_camera_image(sim, env, camera, image_type)
            log(f"cpu_access_end frame={frame} shape={image.shape}")

    gym.destroy_sim(sim)
    log("diagnostic_complete")


if __name__ == "__main__":
    main()
