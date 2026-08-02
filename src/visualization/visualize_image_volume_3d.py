#!/usr/bin/env python3
"""
Headless 3-D visualization for a raw .npy image volume.

This is meant for cluster use: it writes an interactive Plotly HTML file that
you can open in a browser after the script finishes. It does not require Fiji,
ImageJ, X forwarding, or a matplotlib GUI.

The default view is intentionally shell-focused:
    - downsample the raw CT volume
    - threshold at the 95th percentile
    - keep the largest connected bright component
    - render that component as a clean 3-D mesh

Example:
    python src/visualization/visualize_image_volume_3d.py --input volume.npy
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from skimage import measure
from skimage.filters import threshold_otsu


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_ROOT = PROJECT_ROOT / "data" / "visualizations" / "image_volumes"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Write an interactive 3-D HTML visualization for a .npy image volume."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to a 3-D .npy image volume.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output HTML path. Defaults to "
             "data/visualizations/image_volumes/<stem>/<stem>_3d_volume.html.",
    )
    parser.add_argument(
        "--axes",
        default="zyx",
        choices=["zyx", "xyz", "xzy", "yxz", "yzx", "zxy"],
        help="Axis order of the input array. Output scene uses X/Y/Z coordinates.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=4,
        help="Voxel stride for downsampling before 3-D meshing. Larger is faster/smaller.",
    )
    parser.add_argument(
        "--marching-step",
        type=int,
        default=2,
        help="Step size passed to skimage.measure.marching_cubes.",
    )
    parser.add_argument(
        "--threshold",
        default="p95",
        help="Surface threshold. Use a number, 'otsu', or 'pNN' such as p95.",
    )
    parser.add_argument(
        "--max-faces",
        type=int,
        default=220000,
        help="Randomly subsample mesh faces above this count to keep the HTML responsive.",
    )
    parser.add_argument(
        "--mesh-opacity",
        type=float,
        default=0.72,
        help="Surface mesh opacity.",
    )
    parser.add_argument(
        "--with-slices",
        action="store_true",
        help="Also add orthogonal middle-slice planes. Off by default because it can obscure the 3-D shell.",
    )
    parser.add_argument(
        "--slice-opacity",
        type=float,
        default=0.70,
        help="Opacity for orthogonal slice planes.",
    )
    parser.add_argument(
        "--all-components",
        action="store_true",
        help="Render all thresholded components instead of only the largest connected component.",
    )
    parser.add_argument(
        "--min-component-voxels",
        type=int,
        default=0,
        help="When --all-components is used, discard components smaller than this many downsampled voxels.",
    )
    parser.add_argument(
        "--p-low",
        type=float,
        default=0.5,
        help="Lower display percentile for slice contrast.",
    )
    parser.add_argument(
        "--p-high",
        type=float,
        default=99.5,
        help="Upper display percentile for slice contrast.",
    )
    parser.add_argument(
        "--plotlyjs",
        choices=["cdn", "inline"],
        default="inline",
        help="Use 'inline' for a self-contained HTML, or 'cdn' for a smaller file.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when mesh faces are subsampled.",
    )
    return parser.parse_args()


def as_zyx(volume, axes):
    axes = axes.lower()
    order = [axes.index("z"), axes.index("y"), axes.index("x")]
    if order == [0, 1, 2]:
        return volume
    return np.transpose(volume, order)


def parse_threshold(threshold_spec, volume):
    spec = str(threshold_spec).strip().lower()
    if spec == "otsu":
        return float(threshold_otsu(volume))
    if spec.startswith("p"):
        return float(np.percentile(volume, float(spec[1:])))
    return float(spec)


def estimate_contrast(volume, p_low, p_high):
    lo, hi = np.percentile(volume, [p_low, p_high])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.nanmin(volume))
        hi = float(np.nanmax(volume))
    if hi <= lo:
        hi = lo + 1.0
    return float(lo), float(hi)


def add_center_slices(fig, volume, stride, cmin, cmax, opacity):
    nz, ny, nx = volume.shape
    z_mid = nz // 2
    y_mid = ny // 2
    x_mid = nx // 2

    x = np.arange(nx) * stride
    y = np.arange(ny) * stride
    z = np.arange(nz) * stride

    xx, yy = np.meshgrid(x, y)
    fig.add_trace(
        go.Surface(
            x=xx,
            y=yy,
            z=np.full_like(xx, z[z_mid]),
            surfacecolor=volume[z_mid, :, :],
            colorscale="Gray",
            cmin=cmin,
            cmax=cmax,
            opacity=opacity,
            showscale=False,
            name=f"Axial slice Z={z[z_mid]}",
        )
    )

    xx, zz = np.meshgrid(x, z)
    fig.add_trace(
        go.Surface(
            x=xx,
            y=np.full_like(xx, y[y_mid]),
            z=zz,
            surfacecolor=volume[:, y_mid, :],
            colorscale="Gray",
            cmin=cmin,
            cmax=cmax,
            opacity=opacity,
            showscale=False,
            name=f"Coronal slice Y={y[y_mid]}",
        )
    )

    yy, zz = np.meshgrid(y, z)
    fig.add_trace(
        go.Surface(
            x=np.full_like(yy, x[x_mid]),
            y=yy,
            z=zz,
            surfacecolor=volume[:, :, x_mid],
            colorscale="Gray",
            cmin=cmin,
            cmax=cmax,
            opacity=opacity,
            showscale=False,
            name=f"Sagittal slice X={x[x_mid]}",
        )
    )


def component_mask(volume, threshold, all_components, min_component_voxels):
    mask = volume > threshold
    if not mask.any():
        raise ValueError(f"No voxels are above threshold {threshold:.3f}.")
    if mask.all():
        raise ValueError(f"All voxels are above threshold {threshold:.3f}; choose a higher threshold.")

    labels = measure.label(mask, connectivity=1)
    counts = np.bincount(labels.ravel())
    counts[0] = 0

    if all_components:
        keep = counts >= int(min_component_voxels)
        keep[0] = False
        return keep[labels]

    largest = int(np.argmax(counts))
    return labels == largest


def build_mesh(volume, threshold, stride, marching_step, max_faces, seed, all_components, min_component_voxels):
    mask = component_mask(volume, threshold, all_components, min_component_voxels)

    verts_zyx, faces, _normals, _values = measure.marching_cubes(
        mask.astype(np.uint8),
        level=0.5,
        spacing=(stride, stride, stride),
        step_size=max(1, int(marching_step)),
    )

    if max_faces > 0 and len(faces) > max_faces:
        rng = np.random.default_rng(seed)
        keep = rng.choice(len(faces), size=max_faces, replace=False)
        faces = faces[np.sort(keep)]

    return verts_zyx, faces


def main():
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.is_file():
        sys.exit(f"Input volume not found: {input_path}")

    stride = max(1, int(args.stride))
    if args.out:
        out_path = Path(args.out)
    else:
        stem = input_path.stem
        out_path = DEFAULT_OUT_ROOT / stem / f"{stem}_3d_volume.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading volume with memory mapping: {input_path}")
    raw = np.load(input_path, mmap_mode="r")
    if raw.ndim != 3:
        sys.exit(f"Expected a 3-D array, got shape {raw.shape}")

    raw_zyx = as_zyx(raw, args.axes)
    volume = np.asarray(raw_zyx[::stride, ::stride, ::stride])
    print(f"Original shape ZYX: {raw_zyx.shape}")
    print(f"Downsampled shape ZYX: {volume.shape}  (stride={stride})")
    print(f"Volume dtype: {volume.dtype}")

    threshold = parse_threshold(args.threshold, volume)
    cmin, cmax = estimate_contrast(volume, args.p_low, args.p_high)
    print(f"Surface threshold: {threshold:.3f} ({args.threshold})")
    print(f"Slice contrast: cmin={cmin:.3f}, cmax={cmax:.3f}")

    print("Running marching cubes...")
    verts_zyx, faces = build_mesh(
        volume=volume,
        threshold=threshold,
        stride=stride,
        marching_step=args.marching_step,
        max_faces=args.max_faces,
        seed=args.seed,
        all_components=args.all_components,
        min_component_voxels=args.min_component_voxels,
    )
    print(f"Mesh vertices: {len(verts_zyx):,}")
    print(f"Mesh faces: {len(faces):,}")

    fig = go.Figure()
    fig.add_trace(
        go.Mesh3d(
            x=verts_zyx[:, 2],
            y=verts_zyx[:, 1],
            z=verts_zyx[:, 0],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            color="#f2f5f7",
            opacity=args.mesh_opacity,
            name=f"Isosurface > {threshold:.1f}",
            flatshading=False,
            lighting=dict(ambient=0.36, diffuse=0.78, specular=0.26, roughness=0.64, fresnel=0.08),
            lightposition=dict(x=250, y=-320, z=650),
        )
    )

    if args.with_slices:
        print("Adding orthogonal middle slices...")
        add_center_slices(fig, volume, stride, cmin, cmax, args.slice_opacity)

    fig.update_layout(
        title=(
            f"{input_path.stem} raw image volume | "
            f"shape={tuple(raw_zyx.shape)} | stride={stride} | threshold={threshold:.1f}"
        ),
        scene=dict(
            xaxis=dict(title="X", showgrid=False, zeroline=False, showbackground=False),
            yaxis=dict(title="Y", showgrid=False, zeroline=False, showbackground=False),
            zaxis=dict(title="Z", showgrid=False, zeroline=False, showbackground=False),
            aspectmode="data",
            bgcolor="rgb(5, 7, 10)",
            camera=dict(
                eye=dict(x=1.65, y=-1.85, z=1.25),
                up=dict(x=0, y=0, z=1),
            ),
        ),
        legend=dict(bgcolor="rgba(255,255,255,0.80)"),
        margin=dict(l=0, r=0, b=0, t=48),
        paper_bgcolor="white",
    )

    include_plotlyjs = True if args.plotlyjs == "inline" else "cdn"
    print(f"Writing HTML: {out_path}")
    fig.write_html(out_path, include_plotlyjs=include_plotlyjs)
    print("Done.")
    print(f"Open this file in a browser: {out_path}")


if __name__ == "__main__":
    main()
