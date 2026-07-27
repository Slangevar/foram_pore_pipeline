#!/usr/bin/env python3
"""
Chamber Volume Estimation — three assignment methods for comparison.

For each editor-state NPZ the script produces one HTML per method:

  Method A  centroid-knn   (original) — nearest chamber centroid
  Method B  pore-knn       — nearest pore voxel (from pore_voxels in NPZ)
  Method C  flood-fill     — BFS from pore voxels outward through shell (vol==1)

Usage:
    python scripts/analysis/chamber_volume_estimate.py \
        --state-dir data/corrected_cluster_state/all122_addback20_k3_clean/editor_state \
        --volumes MOM_31_03,MOM_250_04,MOM_398_7,MOM_398_3,MOM_890_3 \
        --out-dir data/analysis/chamber_volume_viz \
        --max-vis-points 300000 \
        --methods centroid-knn,pore-knn,flood-fill
"""

import argparse
import glob
import os
import sys
from collections import deque

import numpy as np
from scipy.spatial import cKDTree

try:
    import plotly.graph_objects as go
except ImportError:
    sys.exit("plotly is required: pip install plotly")


# Same PALETTE as cluster_editor_vue.py
PALETTE = [
    '#E6194B', '#3CB44B', '#4363D8', '#F58231', '#911EB4',
    '#42D4F4', '#F032E6', '#BFEF45', '#FABED4', '#469990',
    '#DCBEFF', '#9A6324', '#800000', '#AAFFC3', '#808000',
    '#FFD8B1', '#000075', '#A9A9A9', '#000000', '#E6BEFF',
]


def chamber_color(cid):
    return PALETTE[max(0, int(cid)) % len(PALETTE)]


def load_state(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    return {
        'centroids_3d':     d['centroids_3d'],
        'labels':           d['labels'].copy(),
        'volume_path':      str(d.get('volume_path', '')),
        'pore_voxels':      d['pore_voxels'],       # (N,3) sampled voxel coords
        'pore_voxels_owner': d['pore_voxels_owner'], # (N,)  0-based pore index
    }


# ---------------------------------------------------------------------------
# Method A: centroid-KNN (original baseline)
# ---------------------------------------------------------------------------

def assign_centroid_knn(shell_coords, centroids_3d, labels):
    """Assign each shell voxel to the nearest chamber centroid."""
    unique_cids = np.array(sorted(c for c in set(labels.tolist()) if c >= 2))
    chamber_cents = np.stack([
        centroids_3d[labels == cid].mean(axis=0) for cid in unique_cids
    ])
    tree = cKDTree(chamber_cents)
    _, nn_idx = tree.query(shell_coords, k=1, workers=-1)
    return unique_cids[nn_idx], unique_cids


# ---------------------------------------------------------------------------
# Method B: pore-KNN
# ---------------------------------------------------------------------------

def assign_pore_knn(shell_coords, labels, pore_voxels, pore_voxels_owner):
    """
    Assign each shell voxel to the nearest sampled pore voxel, then look up
    that pore's chamber from labels.

    pore_voxels_owner  — 0-based pore index into the labels array
    labels[pore_idx]   — chamber ID for that pore (≥2 = valid chamber)
    """
    unique_cids = np.array(sorted(c for c in set(labels.tolist()) if c >= 2))

    # Keep only pore voxels whose owner belongs to a valid chamber
    owner_cids = labels[pore_voxels_owner]          # chamber ID for every sampled voxel
    valid_mask = owner_cids >= 2
    pv_coords  = pore_voxels[valid_mask].astype(np.float32)
    pv_cids    = owner_cids[valid_mask]

    tree = cKDTree(pv_coords)
    _, nn_idx = tree.query(shell_coords.astype(np.float32), k=1, workers=-1)
    shell_cids = pv_cids[nn_idx]
    return shell_cids, unique_cids


# ---------------------------------------------------------------------------
# Method C: flood-fill from pore voxels through shell
# ---------------------------------------------------------------------------

_NEIGHBORS_26 = np.array([
    (dz, dy, dx)
    for dz in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dx in (-1, 0, 1)
    if not (dz == 0 and dy == 0 and dx == 0)
], dtype=np.int8)


def assign_flood_fill(vol, shell_coords, labels, pore_voxels, pore_voxels_owner):
    """
    BFS outward from pore voxels through shell voxels (vol==1).

    Seeds: every sampled pore voxel is seeded with its chamber ID.
    Expansion: a shell voxel inherits the chamber ID of the first BFS
    wave that reaches it (ties broken by arrival order, i.e. distance
    from nearest pore voxel).

    Returns shell_cids (N,) aligned with shell_coords.
    """
    unique_cids = np.array(sorted(c for c in set(labels.tolist()) if c >= 2))

    Z, Y, X = vol.shape
    # assignment array: -1 = unassigned; indexed over full volume
    assignment = np.full((Z, Y, X), -1, dtype=np.int32)

    # Seed the queue with pore voxels that belong to valid chambers
    owner_cids = labels[pore_voxels_owner]
    valid_mask = owner_cids >= 2
    seed_coords = pore_voxels[valid_mask]
    seed_cids   = owner_cids[valid_mask]

    queue = deque()
    for (z, y, x), cid in zip(seed_coords, seed_cids):
        z, y, x = int(z), int(y), int(x)
        if assignment[z, y, x] == -1:
            assignment[z, y, x] = int(cid)
            # Only expand into shell voxels from here
            if vol[z, y, x] == 1:
                queue.append((z, y, x))

    print(f'    BFS: seeded {len(seed_coords):,} pore voxels, '
          f'queue starts at {len(queue):,} shell-adjacent seeds', flush=True)

    # BFS expansion — only through shell voxels
    visited = 0
    while queue:
        z, y, x = queue.popleft()
        cid = assignment[z, y, x]
        for dz, dy, dx in _NEIGHBORS_26:
            nz, ny, nx = z + dz, y + dy, x + dx
            if 0 <= nz < Z and 0 <= ny < Y and 0 <= nx < X:
                if vol[nz, ny, nx] == 1 and assignment[nz, ny, nx] == -1:
                    assignment[nz, ny, nx] = cid
                    queue.append((nz, ny, nx))
                    visited += 1

    print(f'    BFS: visited {visited:,} additional shell voxels', flush=True)

    # Gather assignments for shell_coords
    zz, yy, xx = shell_coords[:, 0], shell_coords[:, 1], shell_coords[:, 2]
    shell_cids = assignment[zz, yy, xx]

    # Any unreached shell voxel (inside aperture gap, etc.) falls back to
    # nearest chamber centroid so the result is always fully assigned
    unassigned_mask = shell_cids == -1
    n_unassigned = unassigned_mask.sum()
    if n_unassigned > 0:
        print(f'    BFS: {n_unassigned:,} shell voxels unreached — '
              f'falling back to centroid-KNN for those', flush=True)
        # build centroid fallback just for unassigned voxels
        # (reuse centroids_3d passed in via the outer function — see caller)
        pass  # fallback handled in process_volume

    return shell_cids, unique_cids, unassigned_mask


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def summarise_and_subsample(shell_coords, shell_cids, unique_cids,
                             max_vis_points, rng):
    """Compute per-chamber voxel counts and subsample for visualisation."""
    n_total = len(shell_coords)
    cids_u, counts = np.unique(shell_cids, return_counts=True)
    volumes = {int(c): int(n) for c, n in zip(cids_u, counts) if c >= 2}

    if n_total > max_vis_points:
        idx        = rng.choice(n_total, size=max_vis_points, replace=False)
        vis_coords = shell_coords[idx]
        vis_cids   = shell_cids[idx]
    else:
        vis_coords = shell_coords
        vis_cids   = shell_cids

    return volumes, vis_coords, vis_cids, n_total


def build_figure(name, method_label, volumes, vis_coords, vis_cids,
                 unique_cids, n_total):
    fig   = go.Figure()
    total = sum(volumes.values())

    for cid in unique_cids:
        mask = vis_cids == cid
        if not mask.any():
            continue
        pts   = vis_coords[mask]
        color = chamber_color(cid)
        nvox  = volumes.get(cid, 0)
        pct   = 100.0 * nvox / max(total, 1)
        fig.add_trace(go.Scatter3d(
            x=pts[:, 2].tolist(),
            y=pts[:, 1].tolist(),
            z=pts[:, 0].tolist(),
            mode='markers',
            marker=dict(size=2.0, color=color, opacity=0.50),
            name=f'Ch {cid}  {nvox:,} vox ({pct:.1f}%)',
            legendgroup=f'ch{cid}',
        ))

    fig.update_layout(
        title=dict(
            text=(f'[{method_label}]  {name}'
                  f'  |  {len(unique_cids)} chambers'
                  f'  |  {n_total:,} shell voxels'),
            font=dict(size=14),
        ),
        scene=dict(
            xaxis=dict(title='X', showgrid=False, zeroline=False,
                       backgroundcolor='#f5f7fa'),
            yaxis=dict(title='Y', showgrid=False, zeroline=False,
                       backgroundcolor='#f5f7fa'),
            zaxis=dict(title='Z', showgrid=False, zeroline=False,
                       backgroundcolor='#f5f7fa'),
            bgcolor='white',
            aspectmode='data',
        ),
        legend=dict(itemsizing='constant', font=dict(size=11),
                    bgcolor='rgba(255,255,255,0.88)'),
        margin=dict(l=0, r=0, t=55, b=0),
        paper_bgcolor='white',
    )
    return fig


# ---------------------------------------------------------------------------
# Per-volume processing
# ---------------------------------------------------------------------------

def process_volume(npz_path, out_dir, max_vis_points, rng, methods):
    name = os.path.basename(npz_path).replace('.npz', '')
    print(f'\n[{name}]', flush=True)

    data             = load_state(npz_path)
    centroids_3d     = data['centroids_3d']
    labels           = data['labels']
    volume_path      = data['volume_path']
    pore_voxels      = data['pore_voxels']
    pore_voxels_owner = data['pore_voxels_owner']

    unique_cids = sorted(c for c in set(labels.tolist()) if c >= 2)
    if not unique_cids:
        print('  [SKIP] No chambers found.')
        return None

    needs_volume = 'flood-fill' in methods
    vol = None
    if needs_volume:
        if not volume_path or not os.path.isfile(volume_path):
            print(f'  [ERROR] volume_path not accessible: {volume_path!r}')
            if set(methods) == {'flood-fill'}:
                return None
            methods = [m for m in methods if m != 'flood-fill']
            print('  Skipping flood-fill; continuing with other methods.')
        else:
            print(f'  Loading volume for flood-fill: {volume_path}', flush=True)
            raw = np.load(volume_path)
            vol = raw[0] if raw.ndim == 4 else raw

    # For centroid-knn / pore-knn we need shell_coords but don't need the
    # full volume — derive from the volume only if it was already loaded,
    # otherwise load just enough.
    if vol is None and ('centroid-knn' in methods or 'pore-knn' in methods):
        if not volume_path or not os.path.isfile(volume_path):
            print(f'  [ERROR] volume_path not accessible: {volume_path!r}')
            return None
        print(f'  Loading volume: {volume_path}', flush=True)
        raw = np.load(volume_path)
        vol = raw[0] if raw.ndim == 4 else raw

    shell_coords = np.argwhere(vol == 1)
    n_total      = len(shell_coords)
    print(f'  Shell voxels: {n_total:,}  |  Chambers: {len(unique_cids)}',
          flush=True)

    results_by_method = {}

    for method in methods:
        print(f'  --- {method} ---', flush=True)

        if method == 'centroid-knn':
            shell_cids, ucids = assign_centroid_knn(
                shell_coords, centroids_3d, labels)

        elif method == 'pore-knn':
            shell_cids, ucids = assign_pore_knn(
                shell_coords, labels, pore_voxels, pore_voxels_owner)

        elif method == 'flood-fill':
            shell_cids, ucids, unassigned_mask = assign_flood_fill(
                vol, shell_coords, labels, pore_voxels, pore_voxels_owner)
            # Fallback: centroid-KNN for any voxel the BFS didn't reach
            if unassigned_mask.any():
                fb_cids, _ = assign_centroid_knn(
                    shell_coords[unassigned_mask], centroids_3d, labels)
                shell_cids[unassigned_mask] = fb_cids

        else:
            print(f'  [WARN] Unknown method {method!r}, skipping.')
            continue

        volumes, vis_coords, vis_cids, _ = summarise_and_subsample(
            shell_coords, shell_cids, np.array(unique_cids),
            max_vis_points, rng)

        for cid in unique_cids:
            pct = 100.0 * volumes.get(cid, 0) / max(sum(volumes.values()), 1)
            print(f'    Chamber {cid:3d}: {volumes.get(cid, 0):8,} vox  '
                  f'({pct:5.1f}%)')

        method_tag = method.replace('-', '_')
        fig      = build_figure(name, method, volumes, vis_coords, vis_cids,
                                np.array(unique_cids), n_total)
        out_path = os.path.join(out_dir,
                                f'{name}_{method_tag}_chamber_volume.html')
        fig.write_html(out_path, include_plotlyjs='cdn')
        print(f'  Saved: {out_path}')

        results_by_method[method] = {
            'n_chambers':         len(unique_cids),
            'total_shell_voxels': n_total,
            'per_chamber':        volumes,
        }

    return {'name': name, 'methods': results_by_method}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description='Chamber volume estimation — multi-method comparison'
    )
    ap.add_argument('--state-dir', required=True,
                    help='Folder containing editor-state .npz files')
    ap.add_argument('--volumes', default='',
                    help='Comma-separated base names (without .npz). '
                         'If empty, processes up to --limit files.')
    ap.add_argument('--limit', type=int, default=5,
                    help='Max volumes when --volumes is not given (default: 5)')
    ap.add_argument('--out-dir', default='data/analysis/chamber_volume_viz',
                    help='Output directory for HTML files and summary CSV')
    ap.add_argument('--max-vis-points', type=int, default=300000,
                    help='Max shell voxels to render per volume (default: 300000)')
    ap.add_argument('--methods', default='centroid-knn,pore-knn,flood-fill',
                    help='Comma-separated list of methods to run '
                         '(centroid-knn, pore-knn, flood-fill)')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    methods = [m.strip() for m in args.methods.split(',') if m.strip()]

    if args.volumes.strip():
        names     = [v.strip() for v in args.volumes.split(',') if v.strip()]
        npz_files = [os.path.join(args.state_dir, f'{n}.npz') for n in names]
    else:
        all_files = sorted(glob.glob(os.path.join(args.state_dir, '*.npz')))
        npz_files = all_files[:args.limit]

    print(f'Processing {len(npz_files)} volume(s) with methods: {methods}')

    rng     = np.random.default_rng(42)
    results = []
    for f in npz_files:
        if not os.path.isfile(f):
            print(f'[WARN] File not found: {f}')
            continue
        r = process_volume(f, args.out_dir, args.max_vis_points, rng, methods)
        if r:
            results.append(r)

    csv_path = os.path.join(args.out_dir, 'chamber_volume_summary.csv')
    with open(csv_path, 'w', encoding='utf-8') as fh:
        fh.write('volume,method,n_chambers,total_shell_voxels\n')
        for r in results:
            for method, info in r['methods'].items():
                fh.write(f"{r['name']},{method},"
                         f"{info['n_chambers']},{info['total_shell_voxels']}\n")
    print(f'\nSummary written to: {csv_path}')
    print('Done.')


if __name__ == '__main__':
    main()
