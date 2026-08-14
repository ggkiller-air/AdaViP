# Real-world training adapters

This package is reserved for AdaViP integrations with the pinned VT-DP and RDP
training backbones. It must remain independent of robot control, camera and
sensor publishers, teleoperation, and online deployment code.

The baseline launchers do not require code in this package. Add the VT-DP and
RDP AdaViP observation-encoder adapters here once their formal two-view,
GelSight, and proprioception schemas have been verified.

`offline_image_dataset.py` is a separate deployment-free adapter for the
no-tactile DP sanity check. It is not part of the Table 2 protocol.
