"""Check a prepared Lakh data folder against its manifest; optionally mark it ready.

  python verify_data.py DATA_DIR [--mark]

Recomputes the SHA-256 of train.npz and val.npz, compares them with
manifest.json, and checks each pack's file and group counts. With --mark it
writes DATA_DIR/.ready only if everything matches; the PACE job chain refuses
to start training without that file.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data", type=Path)
    parser.add_argument("--mark", action="store_true", help="write DATA/.ready if every check passes")
    args = parser.parse_args()
    manifest = json.loads((args.data / "manifest.json").read_text())
    problems, lines = [], []
    for name, expected in manifest["packs"].items():
        path = args.data / f"{name}.npz"
        if not path.exists():
            problems.append(f"{path} is missing")
            continue
        digest = sha256(path)
        if digest != expected["sha256"]:
            problems.append(f"{name}.npz hash {digest[:12]} != manifest {expected['sha256'][:12]}")
        with np.load(path, allow_pickle=False) as pack:
            files, groups = len(pack["ids"]), int(pack["meta_group"].max()) + 1
        if (files, groups) != (expected["files"], expected["groups"]):
            problems.append(f"{name}: {files} files / {groups} groups, manifest says "
                            f"{expected['files']} / {expected['groups']}")
        lines.append(f"{name}: {files} files, {groups} groups, sha256 {digest[:12]}")
    print("\n".join(lines))
    if problems:
        print("NOT READY:\n  " + "\n  ".join(problems))
        sys.exit(1)
    print("data matches manifest.json")
    if args.mark:
        stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        (args.data / ".ready").write_text(f"verified {stamp}\n" + "\n".join(lines) + "\n")
        print(f"wrote {args.data / '.ready'}")


if __name__ == "__main__":
    main()
