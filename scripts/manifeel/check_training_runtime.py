"""Fail fast on invalid tempfile/Unix-socket/PyTorch worker IPC configuration."""

import json

from adavip.manifeel.training_runtime import check_dataloader_ipc


if __name__ == "__main__":
    print(json.dumps(check_dataloader_ipc()), flush=True)
