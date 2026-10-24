"""Prepare immutable per-block hashes on CPU before requesting any H200."""

from __future__ import annotations

import argparse
from pathlib import Path

from adavip.manifeel.stage1_streaming_cache import build_block_index


def main() -> None:
    """Validate the full existing cache and generate a separate block index."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    arguments = parser.parse_args()
    build_block_index(arguments.cache_dir, arguments.output, workers=arguments.workers)


if __name__ == "__main__":
    main()
