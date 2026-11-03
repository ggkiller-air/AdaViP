"""Explicit Field+FS shapes, causal timing and sensor-removal protocol."""

import hashlib
import json

IMAGE_KEYS = ("image", "wrist_image")
SENSOR_KEYS = (*IMAGE_KEYS, "marker", "state", "language")
CONDITION_NAMES = ("full", "minus_V1", "minus_V2", "minus_T", "minus_P", "minus_L", "minus_vision")
CACHE_VERSION = "tabero_field_fs_rgb_u8_v1"
DATA_REVISION = "f468387d8bcabdb53e816650e9ae5c2a7661c2fd"


def fingerprint(value: dict) -> str:
    """Stable manifest identity, including source revision and action convention."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
