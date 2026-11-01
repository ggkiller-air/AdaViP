#!/usr/bin/env python3
"""Download only the pinned, complete GelSight simulation release, with hashes."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import threading
import time

import requests

PREFIX = "Simulation/GelSight_mini/"
REVISION = "5fb2757b88826917242adefd9196424ef8cf2d98"
LOCAL = threading.local()


def fetch_file(record: dict, root: Path, endpoint: str, no_proxy: bool = False,
               resolver_proxy: bool = False) -> int:
    """Resume atomically and verify every file against its Git/LFS object."""
    path = root / record["rfilename"]
    path.parent.mkdir(parents=True, exist_ok=True)
    expected = record.get("lfs", {}).get("sha256", record["blobId"])

    def valid(candidate: Path) -> bool:
        if not candidate.exists() or candidate.stat().st_size != record["size"]:
            return False
        digest = hashlib.sha256() if record.get("lfs") else hashlib.sha1()
        if not record.get("lfs"):
            digest.update(b"blob " + str(record["size"]).encode() + b"\0")
        with candidate.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest() == expected

    if valid(path):
        return record["size"]
    if not hasattr(LOCAL, "session"):
        LOCAL.session = requests.Session()
        LOCAL.session.trust_env = not no_proxy
        LOCAL.resolver = requests.Session()
    temporary = path.with_suffix(path.suffix + ".partial")
    for attempt in range(5):
        try:
            url = endpoint.rstrip("/") + "/datasets/FVL-Lab/OpenViTac/resolve/" + REVISION + "/" + record["rfilename"]
            if resolver_proxy:
                response = LOCAL.resolver.get(url, stream=True, allow_redirects=False, timeout=(10, 25))
                while response.is_redirect:
                    from urllib.parse import urljoin, urlparse
                    download_url = urljoin(response.url, response.headers["Location"])
                    response.close()
                    if urlparse(download_url).hostname in ("huggingface.co", "hf-mirror.com"):
                        response = LOCAL.resolver.get(download_url, stream=True, allow_redirects=False, timeout=(10, 25))
                    else:
                        response = LOCAL.session.get(download_url, stream=True, timeout=(10, 40))
                        break
            else:
                response = LOCAL.session.get(url, stream=True, timeout=(20, 90))
            with response:
                response.raise_for_status()
                with temporary.open("wb") as stream:
                    for block in response.iter_content(1024 * 1024):
                        stream.write(block)
            if not valid(temporary):
                raise ValueError("Content hash mismatch: " + record["rfilename"])
            temporary.replace(path)
            return record["size"]
        except Exception as error:
            print(json.dumps({"retry": attempt + 1, "path": record["rfilename"],
                              "error_type": type(error).__name__}), flush=True)
            if attempt == 4:
                raise
            time.sleep(min(10, 2 ** attempt))
    raise RuntimeError("Unreachable")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--inventory", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--endpoint", default="https://huggingface.co")
    parser.add_argument("--no-proxy", action="store_true", help="Bypass inherited proxies for domestic mirrors")
    parser.add_argument("--resolver-proxy", action="store_true", help="Resolve official URLs via proxy, transfer CDN objects directly")
    args = parser.parse_args()
    inventory = json.loads(args.inventory.read_text())
    if inventory["sha"] != REVISION:
        raise ValueError("Unexpected inventory revision")
    records = [r for r in inventory["siblings"] if r["rfilename"].startswith(PREFIX)]
    if len(records) != 6599 or sum(r["size"] for r in records) != 9455749868:
        raise ValueError("Pinned release inventory changed")
    args.root.mkdir(parents=True, exist_ok=True)
    (args.root / "source_inventory.json").write_text(json.dumps({"revision": REVISION, "files": records}, indent=2))
    completed, total, start = 0, 0, time.monotonic()
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = {pool.submit(fetch_file, record, args.root, args.endpoint, args.no_proxy, args.resolver_proxy): record for record in records}
        for future in as_completed(pending):
            try:
                total += future.result()
                completed += 1
            except Exception as error:
                failures.append({"path": pending[future]["rfilename"], "error": repr(error)})
            if (completed + len(failures)) % 100 == 0:
                print(json.dumps({"files": completed, "bytes": total, "failed": len(failures),
                                  "seconds": round(time.monotonic() - start, 1)}), flush=True)
    report = {"revision": REVISION, "files": completed, "bytes": total, "failures": failures}
    (args.root / "download_report.json").write_text(json.dumps(report, indent=2))
    if failures:
        raise RuntimeError("Download failed: " + str(failures[:3]))
    (args.root / "COMPLETE.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
