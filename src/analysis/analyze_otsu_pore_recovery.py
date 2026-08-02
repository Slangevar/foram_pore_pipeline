import argparse
import csv
import glob
import os

import numpy as np
import scipy.ndimage as ndimage
from skimage.filters import threshold_otsu

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_INPUT_GLOB = os.path.join(PROJECT_ROOT, "data", "cleaned_volumes", "*_cleaned.npy")
DEFAULT_CSV_OUT = os.path.join(PROJECT_ROOT, "data", "analysis", "otsu_pore_recovery_stats.csv")
DEFAULT_MD_OUT = os.path.join(PROJECT_ROOT, "data", "analysis", "otsu_pore_recovery_summary.md")

STRUCTURE_6CONN = np.array([
    [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
    [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
    [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
], dtype=np.uint8)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Audit Otsu pore filtering per volume and estimate how many pores would be "
            "added back with a fixed size threshold (default=20 voxels)."
        )
    )
    parser.add_argument(
        "--input-glob",
        default=DEFAULT_INPUT_GLOB,
        help="Glob for source cleaned volumes (default: data/cleaned_volumes/*_cleaned.npy)",
    )
    parser.add_argument(
        "--fixed-threshold",
        type=int,
        default=20,
        help="Fixed pore-size threshold (voxels) for what-if comparison (default: 20)",
    )
    parser.add_argument(
        "--csv-out",
        default=DEFAULT_CSV_OUT,
        help="Output CSV path",
    )
    parser.add_argument(
        "--md-out",
        default=DEFAULT_MD_OUT,
        help="Output Markdown summary path",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional max number of volumes to process (0 means all)",
    )
    return parser.parse_args()


def _extract_volume(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 4:
        return arr[0]
    return arr


def _safe_otsu_threshold(pore_volumes: np.ndarray) -> float:
    valid_volumes = pore_volumes[pore_volumes > 0]
    if valid_volumes.size == 0:
        return 0.0
    log_thresh = threshold_otsu(np.log(valid_volumes))
    return float(np.exp(log_thresh))


def analyze_volume(volume_path: str, fixed_threshold: int) -> dict:
    data = np.load(volume_path)
    vol = _extract_volume(data)

    shell_mask = (vol == 1)
    pore_mask = (vol == 2)

    labeled_pores, num_pores = ndimage.label(pore_mask, structure=STRUCTURE_6CONN)
    if num_pores == 0:
        base = os.path.basename(volume_path).replace("_cleaned.npy", "")
        return {
            "volume": base,
            "num_pores_total": 0,
            "otsu_size_threshold": 0.0,
            "fixed_size_threshold": fixed_threshold,
            "kept_by_otsu_count": 0,
            "removed_by_otsu_count": 0,
            "kept_by_fixed_count": 0,
            "removed_by_fixed_count": 0,
            "added_back_count": 0,
            "added_back_voxels": 0,
            "kept_by_otsu_voxels": 0,
            "removed_by_otsu_voxels": 0,
            "kept_by_fixed_voxels": 0,
            "removed_by_fixed_voxels": 0,
            "spatial_valid_otsu_count": 0,
            "spatial_valid_fixed_count": 0,
            "spatial_added_back_count": 0,
        }

    pore_volumes = ndimage.sum(pore_mask, labeled_pores, range(1, num_pores + 1)).astype(np.float64)
    otsu_threshold = _safe_otsu_threshold(pore_volumes)

    pore_ids = np.arange(1, num_pores + 1)
    keep_otsu = pore_volumes >= otsu_threshold
    keep_fixed = pore_volumes >= fixed_threshold

    kept_otsu_ids = pore_ids[keep_otsu]
    kept_fixed_ids = pore_ids[keep_fixed]

    removed_otsu_ids = pore_ids[~keep_otsu]
    removed_fixed_ids = pore_ids[~keep_fixed]

    # Components recovered when switching from Otsu to a fixed threshold.
    # This means they were removed by Otsu but kept by fixed-threshold filtering.
    added_back_mask = (~keep_otsu) & keep_fixed
    added_back_ids = pore_ids[added_back_mask]

    dilated_shell = ndimage.binary_dilation(shell_mask, iterations=3)

    # Fast component-level shell-touch check using original labels.
    # This avoids relabeling large full-volume masks repeatedly.
    touching_ids = np.unique(labeled_pores[dilated_shell & (labeled_pores > 0)])

    kept_otsu_touching = np.intersect1d(kept_otsu_ids, touching_ids, assume_unique=False)
    kept_fixed_touching = np.intersect1d(kept_fixed_ids, touching_ids, assume_unique=False)
    recovered_spatial_ids = np.intersect1d(added_back_ids, touching_ids, assume_unique=False)

    base = os.path.basename(volume_path).replace("_cleaned.npy", "")
    return {
        "volume": base,
        "num_pores_total": int(num_pores),
        "otsu_size_threshold": float(otsu_threshold),
        "fixed_size_threshold": int(fixed_threshold),
        "kept_by_otsu_count": int(keep_otsu.sum()),
        "removed_by_otsu_count": int((~keep_otsu).sum()),
        "kept_by_fixed_count": int(keep_fixed.sum()),
        "removed_by_fixed_count": int((~keep_fixed).sum()),
        "added_back_count": int(added_back_mask.sum()),
        "added_back_voxels": int(np.round(pore_volumes[added_back_mask].sum())),
        "kept_by_otsu_voxels": int(np.round(pore_volumes[keep_otsu].sum())),
        "removed_by_otsu_voxels": int(np.round(pore_volumes[~keep_otsu].sum())),
        "kept_by_fixed_voxels": int(np.round(pore_volumes[keep_fixed].sum())),
        "removed_by_fixed_voxels": int(np.round(pore_volumes[~keep_fixed].sum())),
        "spatial_valid_otsu_count": int(kept_otsu_touching.size),
        "spatial_valid_fixed_count": int(kept_fixed_touching.size),
        "spatial_added_back_count": int(recovered_spatial_ids.size),
    }


def write_csv(rows, csv_path):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    fieldnames = [
        "volume",
        "num_pores_total",
        "otsu_size_threshold",
        "fixed_size_threshold",
        "kept_by_otsu_count",
        "removed_by_otsu_count",
        "kept_by_fixed_count",
        "removed_by_fixed_count",
        "added_back_count",
        "added_back_voxels",
        "kept_by_otsu_voxels",
        "removed_by_otsu_voxels",
        "kept_by_fixed_voxels",
        "removed_by_fixed_voxels",
        "spatial_valid_otsu_count",
        "spatial_valid_fixed_count",
        "spatial_added_back_count",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows, fixed_threshold):
    if not rows:
        print("No rows to summarize.")
        return

    total_volumes = len(rows)
    total_pores = sum(r["num_pores_total"] for r in rows)
    removed_by_otsu = sum(r["removed_by_otsu_count"] for r in rows)
    added_back_count = sum(r["added_back_count"] for r in rows)
    added_back_voxels = sum(r["added_back_voxels"] for r in rows)
    spatial_added_back = sum(r["spatial_added_back_count"] for r in rows)

    print("================================================")
    print("Otsu Pore Recovery Audit")
    print("================================================")
    print(f"Volumes analyzed: {total_volumes}")
    print(f"Total connected pores: {total_pores}")
    print(f"Total pores removed by Otsu: {removed_by_otsu}")
    print(f"Total pores added back with threshold={fixed_threshold}: {added_back_count}")
    print(f"Total voxels added back with threshold={fixed_threshold}: {added_back_voxels}")
    print(f"Total spatially valid pores added back (after shell-touch check): {spatial_added_back}")

    top_recovered = sorted(rows, key=lambda r: r["added_back_count"], reverse=True)[:10]
    print("\nTop 10 volumes by recovered pore count:")
    for r in top_recovered:
        print(
            f"  {r['volume']}: +{r['added_back_count']} pores "
            f"(+{r['added_back_voxels']} vox), "
            f"Otsu thr={r['otsu_size_threshold']:.2f}"
        )


def write_markdown_summary(rows, fixed_threshold, md_path, csv_path):
    os.makedirs(os.path.dirname(md_path), exist_ok=True)
    if not rows:
        with open(md_path, "w") as f:
            f.write("# Otsu Pore Recovery Audit\n\nNo rows available.\n")
        return

    total_volumes = len(rows)
    total_pores = sum(r["num_pores_total"] for r in rows)
    removed_by_otsu = sum(r["removed_by_otsu_count"] for r in rows)
    added_back_count = sum(r["added_back_count"] for r in rows)
    added_back_voxels = sum(r["added_back_voxels"] for r in rows)
    spatial_added_back = sum(r["spatial_added_back_count"] for r in rows)

    top_recovered = sorted(rows, key=lambda r: r["added_back_count"], reverse=True)[:15]
    top_otsu_removed = sorted(rows, key=lambda r: r["removed_by_otsu_count"], reverse=True)[:15]

    with open(md_path, "w") as f:
        f.write("# Otsu Pore Recovery Audit\n\n")
        f.write("## Global Summary\n")
        f.write(f"- Volumes analyzed: {total_volumes}\n")
        f.write(f"- Total connected pores: {total_pores}\n")
        f.write(f"- Total pores removed by Otsu: {removed_by_otsu}\n")
        f.write(f"- Total pores added back with threshold={fixed_threshold}: {added_back_count}\n")
        f.write(f"- Total voxels added back with threshold={fixed_threshold}: {added_back_voxels}\n")
        f.write(f"- Total spatially valid pores added back (shell-touch): {spatial_added_back}\n\n")

        f.write("## Top Volumes by Recovered Pores\n")
        f.write("| Volume | Added Back Pores | Added Back Voxels | Otsu Threshold |\n")
        f.write("|---|---:|---:|---:|\n")
        for r in top_recovered:
            f.write(
                f"| {r['volume']} | {r['added_back_count']} | {r['added_back_voxels']} | {r['otsu_size_threshold']:.3f} |\n"
            )
        f.write("\n")

        f.write("## Top Volumes by Otsu-Removed Pores\n")
        f.write("| Volume | Otsu Removed Pores | Otsu Removed Voxels | Otsu Threshold |\n")
        f.write("|---|---:|---:|---:|\n")
        for r in top_otsu_removed:
            f.write(
                f"| {r['volume']} | {r['removed_by_otsu_count']} | {r['removed_by_otsu_voxels']} | {r['otsu_size_threshold']:.3f} |\n"
            )
        f.write("\n")

        f.write("## Data Files\n")
        f.write(f"- CSV (per-volume full stats): `{csv_path}`\n")
        f.write(f"- Markdown summary: `{md_path}`\n")


def main():
    args = parse_args()

    paths = sorted(glob.glob(args.input_glob))
    if args.limit and args.limit > 0:
        paths = paths[:args.limit]

    if not paths:
        print(f"No input volumes found for glob: {args.input_glob}")
        return

    rows = []
    for idx, path in enumerate(paths, start=1):
        row = analyze_volume(path, args.fixed_threshold)
        rows.append(row)
        if idx % 10 == 0 or idx == len(paths):
            print(f"Processed {idx}/{len(paths)} volumes...")

    write_csv(rows, args.csv_out)
    print_summary(rows, args.fixed_threshold)
    write_markdown_summary(rows, args.fixed_threshold, args.md_out, args.csv_out)
    print(f"\nCSV written to: {args.csv_out}")
    print(f"Markdown summary written to: {args.md_out}")


if __name__ == "__main__":
    main()
