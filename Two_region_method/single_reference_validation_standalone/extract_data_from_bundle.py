#!/usr/bin/env python3
"""
Optional provenance utility.

Re-create two_region_validation_data.npz directly from MFPT_all_branches.bundle.
This extracts only the raw saved RW P_st and MFPT arrays from the three
TWO-REGION branches. No reconstructed free-energy arrays are copied.

Usage:
    python extract_data_from_bundle.py /path/to/MFPT_all_branches.bundle
"""

import argparse
import io
import subprocess
import tempfile
from pathlib import Path
import numpy as np

CASES = {
    "low": "redo_graph_from_two_region_low",
    "medium": "redo_graph_from_two_region_medium",
    "high": "redo_graph_from_two_region_high",
}

FILES = {
    "PstA": "two_region_Pst_n1.npy",
    "PstB": "two_region_Pst_n2.npy",
    "mfptA": "two_region_mfpt_n1.npy",
    "mfptB": "two_region_mfpt_n2.npy",
}

PREFIX = "Two_region_method/Random_walk_Model/data"


def run(cmd, capture=False):
    kwargs = {"check": True}
    if capture:
        kwargs["stdout"] = subprocess.PIPE
    return subprocess.run(cmd, **kwargs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("bundle", type=Path)
    p.add_argument("--output", type=Path, default=Path("two_region_validation_data.npz"))
    args = p.parse_args()

    bundle = args.bundle.resolve()
    arrays = {}

    with tempfile.TemporaryDirectory(prefix="mfpt_bundle_extract_") as tmp:
        repo = Path(tmp) / "repo"
        run(["git", "-c", "init.defaultBranch=main", "clone", "--quiet", str(bundle), str(repo)])
        run([
            "git", "-C", str(repo), "fetch", "--quiet", str(bundle),
            "refs/remotes/origin/*:refs/remotes/bundle/*",
        ])

        for case, branch in CASES.items():
            for key, filename in FILES.items():
                result = run([
                    "git", "-c", f"safe.directory={repo}", "-C", str(repo),
                    "show", f"remotes/bundle/{branch}:{PREFIX}/{filename}",
                ], capture=True)
                arrays[f"{case}_{key}"] = np.load(io.BytesIO(result.stdout), allow_pickle=False)

    np.savez_compressed(args.output, **arrays)
    print(f"Wrote {args.output.resolve()}")
    print("Keys:")
    for key in sorted(arrays):
        print(f"  {key}: shape={arrays[key].shape}")


if __name__ == "__main__":
    main()
