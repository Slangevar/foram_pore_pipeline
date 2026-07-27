#!/usr/bin/env python3
"""
Reindex chamber clusters by pore count (descending) and mirror the full
corrected-cluster-state folder tree.

Input folder structure (--in-dir  = the source, e.g. all122_addback20_k3_clean):
    <in-dir>/
        editor_state/       *.npz   lightweight cluster state for the Vue editor
        summary_npz/        *.npz   add-back statistics (machine-readable)
        summary_tsv/        *.tsv   add-back statistics (human-readable)
        corrected_volume/   *.npy   full 3-D segmentation volume with chamber labels
        addback_manifest.tsv        one-row-per-volume add-back run manifest

Output folder structure (--out-dir) mirrors the above.  Only the chamber label
IDs are remapped; all voxel coordinates and other data are copied verbatim.

Label convention after reindexing
----------------------------------
  0   deleted pore (preserved as-is, excluded from analysis)
  1   reserved for shell voxels in the 3-D volume encoding
      (never appears in the NPZ labels arrays)
  2   chamber with the MOST pores
  3   chamber with second most pores
  …

Files updated per-volume:
  editor_state/    labels array remapped
  summary_npz/     assigned_labels array remapped
  summary_tsv/     assigned_label column remapped
  corrected_volume/ voxel values ≥2 remapped (shell=1 and background=0 untouched)

Skip flags
----------
  --skip-editor-state   Do not reprocess editor_state/ (e.g. already done).
                        The remap is then derived by comparing old vs new labels
                        from the already-written --out-dir editor_state files.

Usage:
    # Full run (first time):
    python scripts/analysis/reindex_clusters.py \
        --in-dir  data/corrected_cluster_state/all122_addback20_k3_clean \
        --out-dir data/final_foram_state

    # editor_state already done — only process the remaining subfolders:
    python scripts/analysis/reindex_clusters.py \
        --in-dir  data/corrected_cluster_state/all122_addback20_k3_clean \
        --out-dir data/final_foram_state \
        --skip-editor-state
"""

import argparse
import glob
import os
import shutil

import numpy as np


# ---------------------------------------------------------------------------
# Remap helpers
# ---------------------------------------------------------------------------

def build_remap_from_labels(labels: np.ndarray) -> dict:
    """Build {old_cid: new_cid} by sorting chambers by descending pore count."""
    unique_cids = sorted(c for c in set(labels.tolist()) if c >= 2)
    if not unique_cids:
        return {}
    pore_counts = {cid: int((labels == cid).sum()) for cid in unique_cids}
    ranked = sorted(unique_cids, key=lambda c: (-pore_counts[c], c))
    return {old: (i + 2) for i, old in enumerate(ranked)}


def build_remap_from_pair(old_labels: np.ndarray, new_labels: np.ndarray) -> dict:
    """
    Reconstruct the remap by comparing old and new label arrays element-wise.
    Used when editor_state has already been reindexed.
    """
    remap = {}
    for old, new in zip(old_labels.tolist(), new_labels.tolist()):
        if old >= 2 and new >= 2 and old not in remap:
            remap[old] = new
    return remap


def apply_remap(arr: np.ndarray, remap: dict) -> np.ndarray:
    """Return a copy of arr with integer values remapped; unknowns kept as-is."""
    out = arr.copy()
    for old, new in remap.items():
        out[arr == old] = new
    return out


# ---------------------------------------------------------------------------
# Per-subfolder handlers
# ---------------------------------------------------------------------------

def process_editor_state(base: str, in_dir: str, out_dir: str) -> dict:
    src = os.path.join(in_dir, f"{base}.npz")
    dst = os.path.join(out_dir, f"{base}.npz")
    d      = np.load(src, allow_pickle=True)
    labels = d["labels"].copy()
    remap  = build_remap_from_labels(labels)

    arrays = {k: d[k] for k in d.files}
    if remap:
        arrays["labels"] = apply_remap(labels, remap)
    np.savez_compressed(dst.replace(".npz", ""), **arrays)

    n = len(remap)
    print(f"    [editor_state]   {base}: {n} chambers remapped")
    return remap


def derive_remap_from_existing(base: str, in_dir: str, out_dir: str) -> dict:
    """Read old and already-written new labels to reconstruct the remap."""
    src_old = os.path.join(in_dir, f"{base}.npz")
    src_new = os.path.join(out_dir, f"{base}.npz")
    if not os.path.isfile(src_new):
        print(f"    [editor_state]   {base}: reindexed file missing in out-dir, "
              "deriving remap from source")
        d = np.load(src_old, allow_pickle=True)
        return build_remap_from_labels(d["labels"])
    old_labels = np.load(src_old, allow_pickle=True)["labels"]
    new_labels = np.load(src_new, allow_pickle=True)["labels"]
    remap = build_remap_from_pair(old_labels, new_labels)
    print(f"    [editor_state]   {base}: remap derived from existing file "
          f"({len(remap)} chambers)")
    return remap


def process_summary_npz(base: str, in_dir: str, out_dir: str, remap: dict):
    src = os.path.join(in_dir, f"{base}.npz")
    dst = os.path.join(out_dir, f"{base}.npz")
    if not os.path.isfile(src):
        return
    d      = np.load(src, allow_pickle=True)
    arrays = {k: d[k] for k in d.files}
    if remap and "assigned_labels" in arrays:
        arrays["assigned_labels"] = apply_remap(
            arrays["assigned_labels"].astype(np.int32), remap
        )
    np.savez_compressed(dst.replace(".npz", ""), **arrays)
    print(f"    [summary_npz]    {base}: assigned_labels remapped")


def process_summary_tsv(base: str, in_dir: str, out_dir: str, remap: dict):
    src = os.path.join(in_dir, f"{base}.tsv")
    dst = os.path.join(out_dir, f"{base}.tsv")
    if not os.path.isfile(src):
        return
    with open(src, encoding="utf-8") as fh:
        lines = fh.readlines()
    if not remap or len(lines) < 2:
        shutil.copy2(src, dst)
        return
    header = lines[0].rstrip("\n").split("\t")
    try:
        col_idx = header.index("assigned_label")
    except ValueError:
        shutil.copy2(src, dst)
        return
    out_lines = [lines[0]]
    for line in lines[1:]:
        parts = line.rstrip("\n").split("\t")
        if len(parts) > col_idx:
            try:
                old = int(parts[col_idx])
                parts[col_idx] = str(remap.get(old, old))
            except ValueError:
                pass
        out_lines.append("\t".join(parts) + "\n")
    with open(dst, "w", encoding="utf-8") as fh:
        fh.writelines(out_lines)
    print(f"    [summary_tsv]    {base}: assigned_label column remapped")


def process_corrected_volume(base: str, in_dir: str, out_dir: str, remap: dict):
    src = os.path.join(in_dir, f"{base}.npy")
    dst = os.path.join(out_dir, f"{base}.npy")
    if not os.path.isfile(src):
        return
    if not remap:
        shutil.copy2(src, dst)
        return
    vol = np.load(src)
    np.save(dst, apply_remap(vol, remap))
    print(f"    [corrected_vol]  {base}: voxel labels remapped")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Mirror corrected-cluster-state folder with reindexed chamber labels"
    )
    ap.add_argument("--in-dir",  required=True,
                    help="Source folder (contains editor_state/, summary_npz/, etc.)")
    ap.add_argument("--out-dir", required=True,
                    help="Destination folder (same subfolder structure will be created)")
    ap.add_argument("--limit", type=int, default=0,
                    help="Process at most N volumes (0 = all)")
    ap.add_argument("--skip-editor-state", action="store_true",
                    help="Skip reprocessing editor_state/ and derive remap from "
                         "already-written files in --out-dir/editor_state/")
    args = ap.parse_args()

    in_editor   = os.path.join(args.in_dir,  "editor_state")
    in_sum_npz  = os.path.join(args.in_dir,  "summary_npz")
    in_sum_tsv  = os.path.join(args.in_dir,  "summary_tsv")
    in_vol      = os.path.join(args.in_dir,  "corrected_volume")

    out_editor  = os.path.join(args.out_dir, "editor_state")
    out_sum_npz = os.path.join(args.out_dir, "summary_npz")
    out_sum_tsv = os.path.join(args.out_dir, "summary_tsv")
    out_vol     = os.path.join(args.out_dir, "clustered_volume")

    for d in (out_editor, out_sum_npz, out_sum_tsv, out_vol):
        os.makedirs(d, exist_ok=True)

    # Copy top-level manifest unchanged
    manifest_src = os.path.join(args.in_dir,  "addback_manifest.tsv")
    manifest_dst = os.path.join(args.out_dir, "addback_manifest.tsv")
    if os.path.isfile(manifest_src):
        shutil.copy2(manifest_src, manifest_dst)
        print(f"Copied addback_manifest.tsv")

    npz_files = sorted(glob.glob(os.path.join(in_editor, "*.npz")))
    if args.limit > 0:
        npz_files = npz_files[: args.limit]

    mode = "skip-editor-state" if args.skip_editor_state else "full"
    print(f"\nProcessing {len(npz_files)} volume(s)  [{mode}]"
          f"  {args.in_dir} → {args.out_dir}\n")

    summary_rows = []
    for npz_path in npz_files:
        base = os.path.basename(npz_path).replace(".npz", "")
        print(f"[{base}]", flush=True)

        if args.skip_editor_state:
            remap = derive_remap_from_existing(base, in_editor, out_editor)
        else:
            remap = process_editor_state(base, in_editor, out_editor)

        process_summary_npz(base, in_sum_npz, out_sum_npz, remap)
        process_summary_tsv(base, in_sum_tsv, out_sum_tsv, remap)
        process_corrected_volume(base, in_vol, out_vol, remap)

        summary_rows.append({
            "volume":     base,
            "n_chambers": len(remap),
            "remapped":   bool(remap),
        })

    csv_path = os.path.join(args.out_dir, "reindex_summary.csv")
    with open(csv_path, "w", encoding="utf-8") as fh:
        fh.write("volume,n_chambers,remapped\n")
        for r in summary_rows:
            fh.write(f"{r['volume']},{r['n_chambers']},{r['remapped']}\n")

    print(f"\nSummary written to: {csv_path}")
    print(f"Done. Processed {len(summary_rows)} volume(s).")


if __name__ == "__main__":
    main()
