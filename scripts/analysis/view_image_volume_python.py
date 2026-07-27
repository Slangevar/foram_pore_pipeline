#!/usr/bin/env python3
"""
Python-native viewer for a 3-D .npy image volume.

Default target:
    data/image_volumes/MOM_7_01.npy

Usage:
    python scripts/analysis/view_image_volume_python.py

In a local Python session this opens an interactive matplotlib window with
axial, coronal, and sagittal slice sliders. On a headless server, use
--preview-only to write PNG previews instead.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "image_volumes" / "MOM_7_01.npy"
DEFAULT_OUT_ROOT = PROJECT_ROOT / "data" / "visualizations" / "image_volumes"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Open a 3-D .npy image volume in a Python/matplotlib slice viewer."
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help="Path to a 3-D .npy image volume. Defaults to MOM_7_01.npy.",
    )
    parser.add_argument(
        "--axes",
        default="zyx",
        choices=["zyx", "xyz", "xzy", "yxz", "yzx", "zxy"],
        help="Axis order of the input array. Viewer internally uses ZYX.",
    )
    parser.add_argument(
        "--out-root",
        default=str(DEFAULT_OUT_ROOT),
        help="Root folder used by --preview-only or --save-previews.",
    )
    parser.add_argument(
        "--cmap",
        default="gray",
        help="Matplotlib colormap name.",
    )
    parser.add_argument(
        "--p-low",
        type=float,
        default=0.5,
        help="Lower display percentile.",
    )
    parser.add_argument(
        "--p-high",
        type=float,
        default=99.5,
        help="Upper display percentile.",
    )
    parser.add_argument(
        "--sample-step",
        type=int,
        default=8,
        help="Stride used for fast contrast percentile estimation.",
    )
    parser.add_argument(
        "--preview-only",
        action="store_true",
        help="Write middle-slice and maximum-intensity projection PNGs, then exit.",
    )
    parser.add_argument(
        "--save-previews",
        action="store_true",
        help="Write preview PNGs before opening the interactive viewer.",
    )
    return parser.parse_args()


def as_zyx(volume, axes):
    axes = axes.lower()
    order = [axes.index("z"), axes.index("y"), axes.index("x")]
    if order == [0, 1, 2]:
        return volume
    return np.transpose(volume, order)


def estimate_contrast(volume, p_low, p_high, sample_step):
    step = max(1, int(sample_step))
    sample = np.asarray(volume[::step, ::step, ::step])
    lo, hi = np.percentile(sample, [p_low, p_high])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.nanmin(sample))
        hi = float(np.nanmax(sample))
    if hi <= lo:
        hi = lo + 1.0
    return float(lo), float(hi)


def save_preview_pngs(volume, out_dir, stem, cmap, vmin, vmax, force_agg=False):
    if force_agg:
        import matplotlib

        matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    z_mid = volume.shape[0] // 2
    y_mid = volume.shape[1] // 2
    x_mid = volume.shape[2] // 2

    previews = {
        "middle_z": volume[z_mid, :, :],
        "middle_y": volume[:, y_mid, :],
        "middle_x": volume[:, :, x_mid],
        "mip_z": np.max(volume, axis=0),
        "mip_y": np.max(volume, axis=1),
        "mip_x": np.max(volume, axis=2),
    }

    written = []
    for label, image in previews.items():
        path = out_dir / f"{stem}_{label}.png"
        plt.imsave(path, image, cmap=cmap, vmin=vmin, vmax=vmax)
        written.append(path)
    return written


class OrthogonalSliceViewer:
    def __init__(self, volume, name, cmap, vmin, vmax):
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Slider

        self.plt = plt
        self.Slider = Slider
        self.volume = volume
        self.name = name
        self.cmap = cmap
        self.vmin = vmin
        self.vmax = vmax
        self.z = volume.shape[0] // 2
        self.y = volume.shape[1] // 2
        self.x = volume.shape[2] // 2

    def show(self):
        v = self.volume
        fig, axes = self.plt.subplots(1, 3, figsize=(14, 6))
        manager = getattr(fig.canvas, "manager", None)
        if manager is not None and hasattr(manager, "set_window_title"):
            manager.set_window_title(f"Volume viewer: {self.name}")
        self.plt.subplots_adjust(left=0.06, right=0.98, bottom=0.22, top=0.88, wspace=0.18)

        im_z = axes[0].imshow(v[self.z, :, :], cmap=self.cmap, vmin=self.vmin, vmax=self.vmax)
        im_y = axes[1].imshow(v[:, self.y, :], cmap=self.cmap, vmin=self.vmin, vmax=self.vmax, aspect="auto")
        im_x = axes[2].imshow(v[:, :, self.x], cmap=self.cmap, vmin=self.vmin, vmax=self.vmax, aspect="auto")

        axes[0].set_title(f"Axial Z={self.z}")
        axes[1].set_title(f"Coronal Y={self.y}")
        axes[2].set_title(f"Sagittal X={self.x}")
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])

        cbar = fig.colorbar(im_z, ax=axes, shrink=0.72, pad=0.015)
        cbar.set_label("Intensity")

        ax_z = fig.add_axes([0.12, 0.13, 0.78, 0.025])
        ax_y = fig.add_axes([0.12, 0.09, 0.78, 0.025])
        ax_x = fig.add_axes([0.12, 0.05, 0.78, 0.025])

        slider_z = self.Slider(ax_z, "Z", 0, v.shape[0] - 1, valinit=self.z, valstep=1)
        slider_y = self.Slider(ax_y, "Y", 0, v.shape[1] - 1, valinit=self.y, valstep=1)
        slider_x = self.Slider(ax_x, "X", 0, v.shape[2] - 1, valinit=self.x, valstep=1)

        def update(_):
            self.z = int(slider_z.val)
            self.y = int(slider_y.val)
            self.x = int(slider_x.val)
            im_z.set_data(v[self.z, :, :])
            im_y.set_data(v[:, self.y, :])
            im_x.set_data(v[:, :, self.x])
            axes[0].set_title(f"Axial Z={self.z}")
            axes[1].set_title(f"Coronal Y={self.y}")
            axes[2].set_title(f"Sagittal X={self.x}")
            fig.canvas.draw_idle()

        slider_z.on_changed(update)
        slider_y.on_changed(update)
        slider_x.on_changed(update)

        fig.suptitle(f"{self.name}  |  shape ZYX={v.shape}  |  dtype={v.dtype}", fontsize=12)
        self.plt.show()


def main():
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.is_file():
        sys.exit(f"Input volume not found: {input_path}")

    if not os.environ.get("MPLCONFIGDIR"):
        os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

    print(f"Loading volume with memory mapping: {input_path}")
    volume = np.load(input_path, mmap_mode="r")
    if volume.ndim != 3:
        sys.exit(f"Expected a 3-D array, got shape {volume.shape}")
    volume = as_zyx(volume, args.axes)

    vmin, vmax = estimate_contrast(volume, args.p_low, args.p_high, args.sample_step)
    stem = input_path.stem
    out_dir = Path(args.out_root) / stem

    print(f"Volume shape ZYX: {volume.shape}")
    print(f"Volume dtype: {volume.dtype}")
    print(f"Display contrast: vmin={vmin:.3f}, vmax={vmax:.3f}")

    if args.preview_only or args.save_previews:
        print(f"Writing preview PNGs to: {out_dir}")
        paths = save_preview_pngs(
            volume,
            out_dir,
            stem,
            args.cmap,
            vmin,
            vmax,
            force_agg=args.preview_only,
        )
        for path in paths:
            print(f"  {path}")

    if args.preview_only:
        return

    import matplotlib

    backend = matplotlib.get_backend().lower()
    if "agg" in backend:
        print(
            "Current matplotlib backend is non-interactive. "
            "Run with a GUI backend, or use --preview-only on this machine."
        )
        return

    viewer = OrthogonalSliceViewer(volume, stem, args.cmap, vmin, vmax)
    viewer.show()


if __name__ == "__main__":
    main()
