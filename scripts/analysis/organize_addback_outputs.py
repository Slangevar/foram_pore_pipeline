import argparse
import glob
import os
import re
import shutil
from typing import Dict

import numpy as np


FLAT_STATE_RE = re.compile(r"^(?P<base>.+)_cluster_state_addback\d+_k\d+\.npz$")
FLAT_SUM_NPZ_RE = re.compile(r"^(?P<base>.+)_addback\d+_k\d+_summary\.npz$")
FLAT_SUM_TSV_RE = re.compile(r"^(?P<base>.+)_addback\d+_k\d+_summary\.tsv$")
FLAT_CORR_RE = re.compile(r"^(?P<base>.+)_addback\d+_k\d+_corrected\.npy$")


def ensure_dirs(root):
    layout = {
        "editor_state": os.path.join(root, "editor_state"),
        "summary_npz": os.path.join(root, "summary_npz"),
        "summary_tsv": os.path.join(root, "summary_tsv"),
        "corrected_volume": os.path.join(root, "corrected_volume"),
    }
    for p in layout.values():
        os.makedirs(p, exist_ok=True)
    return layout


def move_or_copy(src, dst, copy_mode):
    if copy_mode:
        shutil.copy2(src, dst)
    else:
        shutil.move(src, dst)


def _update_if_newer(store: Dict[str, str], base: str, src: str) -> None:
    prev = store.get(base)
    if prev is None or os.path.getmtime(src) >= os.path.getmtime(prev):
        store[base] = src


def _collect_flat(source_dir: str) -> Dict[str, Dict[str, str]]:
    found: Dict[str, Dict[str, str]] = {
        "editor_state": {},
        "summary_npz": {},
        "summary_tsv": {},
        "corrected_volume": {},
    }

    for p in sorted(glob.glob(os.path.join(source_dir, "*.npz"))):
        name = os.path.basename(p)
        m_state = FLAT_STATE_RE.match(name)
        m_sum = FLAT_SUM_NPZ_RE.match(name)
        if m_state:
            _update_if_newer(found["editor_state"], m_state.group("base"), p)
        elif m_sum:
            _update_if_newer(found["summary_npz"], m_sum.group("base"), p)

    for p in sorted(glob.glob(os.path.join(source_dir, "*.tsv"))):
        m = FLAT_SUM_TSV_RE.match(os.path.basename(p))
        if m:
            _update_if_newer(found["summary_tsv"], m.group("base"), p)

    for p in sorted(glob.glob(os.path.join(source_dir, "*.npy"))):
        m = FLAT_CORR_RE.match(os.path.basename(p))
        if m:
            _update_if_newer(found["corrected_volume"], m.group("base"), p)

    return found


def _collect_structured(source_dir: str) -> Dict[str, Dict[str, str]]:
    found: Dict[str, Dict[str, str]] = {
        "editor_state": {},
        "summary_npz": {},
        "summary_tsv": {},
        "corrected_volume": {},
    }

    patterns = {
        "editor_state": "editor_state/*.npz",
        "summary_npz": "summary_npz/*.npz",
        "summary_tsv": "summary_tsv/*.tsv",
        "corrected_volume": "corrected_volume/*.npy",
    }
    for key, rel_pattern in patterns.items():
        for p in sorted(glob.glob(os.path.join(source_dir, rel_pattern))):
            base = os.path.splitext(os.path.basename(p))[0]
            _update_if_newer(found[key], base, p)

    return found


def collect_artifacts(source_dir: str) -> Dict[str, Dict[str, str]]:
    merged: Dict[str, Dict[str, str]] = {
        "editor_state": {},
        "summary_npz": {},
        "summary_tsv": {},
        "corrected_volume": {},
    }
    for source in (_collect_flat(source_dir), _collect_structured(source_dir)):
        for key, mapping in source.items():
            for base, path in mapping.items():
                _update_if_newer(merged[key], base, path)
    return merged


def organize_flat_outputs(source_dir, out_root, copy_mode=False):
    layout = ensure_dirs(out_root)
    artifacts = collect_artifacts(source_dir)

    moved = {"editor_state": 0, "summary_npz": 0, "summary_tsv": 0, "corrected_volume": 0}

    exts = {
        "editor_state": ".npz",
        "summary_npz": ".npz",
        "summary_tsv": ".tsv",
        "corrected_volume": ".npy",
    }
    for key in ("editor_state", "summary_npz", "summary_tsv", "corrected_volume"):
        for base, src in sorted(artifacts[key].items()):
            dst = os.path.join(layout[key], f"{base}{exts[key]}")
            move_or_copy(src, dst, copy_mode)
            moved[key] += 1

    return moved


def write_manifest(out_root):
    summary_dir = os.path.join(out_root, "summary_npz")
    rows = []
    for p in sorted(glob.glob(os.path.join(summary_dir, "*.npz"))):
        base = os.path.basename(p).replace(".npz", "")
        d = np.load(p, allow_pickle=True)
        rows.append((
            base,
            int(d["existing_pores"]),
            int(d["added_pores"]),
            int(d["final_pores"]),
        ))

    manifest = os.path.join(out_root, "addback_manifest.tsv")
    with open(manifest, "w", encoding="utf-8") as f:
        f.write("volume\texisting_pores\tadded_pores\tfinal_pores\n")
        for r in rows:
            f.write(f"{r[0]}\t{r[1]}\t{r[2]}\t{r[3]}\n")

    return manifest, len(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Organize add-back outputs into clean subfolders and short editor filenames."
    )
    parser.add_argument("source_dir", help="Input directory containing add-back outputs")
    parser.add_argument(
        "--out-root",
        default="",
        help="Output root directory (default: <source_dir>_organized)",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy instead of move (keeps source untouched)",
    )
    args = parser.parse_args()

    source_dir = args.source_dir
    out_root = args.out_root or (source_dir.rstrip("/") + "_organized")
    os.makedirs(out_root, exist_ok=True)

    moved = organize_flat_outputs(source_dir, out_root, copy_mode=args.copy)
    manifest, n = write_manifest(out_root)

    print(f"Source: {source_dir}")
    print(f"Organized root: {out_root}")
    print(f"editor_state files: {moved['editor_state']}")
    print(f"summary_npz files: {moved['summary_npz']}")
    print(f"summary_tsv files: {moved['summary_tsv']}")
    print(f"corrected_volume files: {moved['corrected_volume']}")
    print(f"Manifest: {manifest}")
    print(f"Manifest rows: {n}")


if __name__ == "__main__":
    main()
