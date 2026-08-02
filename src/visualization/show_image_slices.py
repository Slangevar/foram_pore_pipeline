#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def save_axis_sheet(vol, axis, out_path, n=16, p_low=0.5, p_high=99.5, cmap="gray"):
    idxs = np.linspace(0, vol.shape[axis] - 1, n, dtype=int)
    sample = np.asarray(vol[::8, ::8, ::8])
    vmin, vmax = np.percentile(sample, [p_low, p_high])

    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows), constrained_layout=True)
    axes = np.asarray(axes).ravel()

    for ax, idx in zip(axes, idxs):
        if axis == 0:
            img = vol[idx, :, :]
            title = f"axis0 z? = {idx}"
        elif axis == 1:
            img = vol[:, idx, :]
            title = f"axis1 y? = {idx}"
        else:
            img = vol[:, :, idx]
            title = f"axis2 x? = {idx}"
        ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    for ax in axes[len(idxs):]:
        ax.axis("off")

    fig.suptitle(f"{out_path.stem} | volume shape={vol.shape} | display p{p_low}-p{p_high}", fontsize=12)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Write contact sheets of raw .npy volume slices.")
    ap.add_argument("--input", required=True, help="Path to a 3-D .npy image volume.")
    ap.add_argument("--out-dir", default=None,
                    help="Output folder. Defaults to "
                         "data/visualizations/image_volumes/<stem>/slices.")
    ap.add_argument("--n", type=int, default=16, help="Number of slices per axis sheet.")
    args = ap.parse_args()

    inp = Path(args.input)
    out_dir = (Path(args.out_dir) if args.out_dir
               else Path("data/visualizations/image_volumes") / inp.stem / "slices")
    out_dir.mkdir(parents=True, exist_ok=True)

    vol = np.load(inp, mmap_mode="r")
    if vol.ndim != 3:
        raise ValueError(f"Expected 3D volume, got shape {vol.shape}")

    print(f"Loaded {inp}: shape={vol.shape}, dtype={vol.dtype}")
    for axis in (0, 1, 2):
        out = out_dir / f"{inp.stem}_axis{axis}_slices.png"
        save_axis_sheet(vol, axis, out, n=args.n)
        print(out)


if __name__ == "__main__":
    main()
