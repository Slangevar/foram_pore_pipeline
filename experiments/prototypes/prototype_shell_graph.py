"""
Shell-Aware Graph Clustering for Foraminifera Pore-Chamber Assignment.

Algorithm:
1. Extract valid pore centroids (Otsu filtering + spatial validation).
2. Build a K-Nearest-Neighbor graph of the 3D centroids.
3. For each edge, sample points along the 3D line between the two centroids.
4. If ANY sampled point falls inside a shell voxel, REMOVE that edge
   (the shell wall physically separates these pores).
5. Connected components of the pruned graph = distinct chambers.

References:
- KNN graphs: standard in spectral/graph-based clustering
- Line-of-sight testing through binary masks: visibility graphs (De Berg et al.),
  used in robotics, medical imaging (vessel segmentation), and GIS
- Connected components: fundamental graph theory
"""

import numpy as np
import scipy.ndimage as ndimage
from scipy.spatial import cKDTree
from skimage.filters import threshold_otsu
import plotly.graph_objects as go
import plotly.express as px
import argparse
import sys
import time


def line_crosses_shell(p1, p2, shell_mask, n_samples=20):
    """
    Sample n_samples points along the 3D line from p1 to p2.
    Return True if ANY sampled point falls inside a shell voxel.
    """
    shape = np.array(shell_mask.shape)
    for t in np.linspace(0, 1, n_samples):
        point = (1 - t) * p1 + t * p2
        # Round to nearest voxel
        idx = np.round(point).astype(int)
        # Bounds check
        if np.any(idx < 0) or np.any(idx >= shape):
            continue
        if shell_mask[idx[0], idx[1], idx[2]]:
            return True
    return False


def process_volume(volume_path, output_html, k_neighbors=12, n_line_samples=25,
                   min_chamber_pores=5):
    print(f"Loading {volume_path}...")
    try:
        data = np.load(volume_path)
    except Exception as e:
        print(f"Error loading {volume_path}: {e}")
        sys.exit(1)

    if data.ndim == 4:
        data = data[0]

    print(f"Volume Shape: {data.shape}")

    shell_mask = (data == 1)
    pore_mask = (data == 2)

    # ── 1. Pore Noise Removal (Otsu Thresholding) ──
    print("Step 1: Extracting valid pores (Otsu + Spatial Validation)...")
    structure_6conn = np.array([
        [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
        [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
        [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
    ])
    labeled_pores, num_pores = ndimage.label(pore_mask, structure=structure_6conn)
    if num_pores == 0:
        print("No pores found!")
        sys.exit(0)

    pore_volumes = ndimage.sum(pore_mask, labeled_pores, range(1, num_pores + 1))
    valid_volumes = pore_volumes[pore_volumes > 0]

    size_threshold = 0
    if len(valid_volumes) > 0:
        log_vols = np.log(valid_volumes)
        try:
            log_thresh = threshold_otsu(log_vols)
            size_threshold = np.exp(log_thresh)
            pores_to_keep = np.where(pore_volumes >= size_threshold)[0] + 1
            print(f"  Otsu threshold: {size_threshold:.1f} voxels. "
                  f"Kept {len(pores_to_keep)} / {num_pores} pores.")
        except Exception:
            pores_to_keep = np.arange(1, num_pores + 1)
    else:
        pores_to_keep = np.arange(1, num_pores + 1)

    # Spatial validation: pores must touch the dilated shell
    dilated_shell = ndimage.binary_dilation(shell_mask, iterations=3)
    filtered_pore_mask = np.isin(labeled_pores, pores_to_keep) & dilated_shell

    valid_labeled_pores, n_valid = ndimage.label(filtered_pore_mask, structure=structure_6conn)
    print(f"  Spatially valid pores: {n_valid}")

    if n_valid < 3:
        print("Too few pores to cluster.")
        sys.exit(0)

    # ── 2. Extract 3D Centroids ──
    centroids = np.array(ndimage.center_of_mass(
        filtered_pore_mask, valid_labeled_pores, range(1, n_valid + 1)
    ))
    print(f"Step 2: Extracted {len(centroids)} pore centroids.")

    # ── 3. Build KNN Graph ──
    k = min(k_neighbors, n_valid - 1)
    print(f"Step 3: Building KNN graph (k={k})...")
    tree = cKDTree(centroids)
    distances, indices = tree.query(centroids, k=k + 1)  # +1 because self is included

    # ── 4. Prune Edges That Cross the Shell Wall ──
    print(f"Step 4: Pruning edges that cross through shell wall "
          f"({n_line_samples} samples per edge)...")
    t0 = time.time()

    # Build adjacency list
    adjacency = {i: set() for i in range(n_valid)}
    edges_total = 0
    edges_pruned = 0

    for i in range(n_valid):
        for j_idx in range(1, k + 1):  # skip index 0 (self)
            j = indices[i, j_idx]
            if j <= i:
                continue  # avoid double-checking symmetric edges
            edges_total += 1

            if line_crosses_shell(centroids[i], centroids[j], shell_mask,
                                  n_samples=n_line_samples):
                edges_pruned += 1
            else:
                adjacency[i].add(j)
                adjacency[j].add(i)

    elapsed = time.time() - t0
    print(f"  Checked {edges_total} edges in {elapsed:.1f}s. "
          f"Pruned {edges_pruned} ({100 * edges_pruned / max(1, edges_total):.1f}%) "
          f"that crossed shell walls.")

    # ── 5. Connected Components = Chambers ──
    print("Step 5: Finding connected components...")
    labels = np.full(n_valid, -1, dtype=int)
    current_label = 0
    for start in range(n_valid):
        if labels[start] >= 0:
            continue
        # BFS
        queue = [start]
        labels[start] = current_label
        while queue:
            node = queue.pop(0)
            for neighbor in adjacency[node]:
                if labels[neighbor] < 0:
                    labels[neighbor] = current_label
                    queue.append(neighbor)
        current_label += 1

    # Merge tiny components (< min_chamber_pores) into noise label -1
    unique_labels, counts = np.unique(labels, return_counts=True)
    for lbl, cnt in zip(unique_labels, counts):
        if cnt < min_chamber_pores:
            labels[labels == lbl] = -1

    # Re-index remaining labels to be consecutive
    remaining = sorted(set(labels) - {-1})
    label_map = {old: new for new, old in enumerate(remaining)}
    label_map[-1] = -1
    labels = np.array([label_map[l] for l in labels])

    num_chambers = len(remaining)
    noise_count = np.sum(labels == -1)
    print(f"  Found {num_chambers} chambers! "
          f"({noise_count} pores in tiny fragments marked as noise)")

    # ── 6. Visualization ──
    print("\nStep 6: Building HTML Visualizer...")
    colors = px.colors.qualitative.Alphabet + px.colors.qualitative.Dark24
    scatters = []

    # Faint shell context
    shell_coords = np.column_stack(np.where(shell_mask))
    if len(shell_coords) > 50000:
        np.random.seed(42)
        idx = np.random.choice(len(shell_coords), 50000, replace=False)
        shell_coords = shell_coords[idx]
    scatters.append(
        go.Scatter3d(
            x=shell_coords[:, 1], y=shell_coords[:, 2], z=shell_coords[:, 0],
            mode='markers',
            marker=dict(size=1, color='white', opacity=0.03),
            name="Shell Outline"
        )
    )

    # Colored pores per chamber
    for cluster_id in sorted(set(labels)):
        if cluster_id == -1:
            name = "Noise / Tiny Fragments"
            color = "gray"
            size = 2
            opacity = 0.3
        else:
            name = f"Chamber {cluster_id}"
            color = colors[cluster_id % len(colors)]
            size = 3
            opacity = 1.0

        pore_label_indices = np.where(labels == cluster_id)[0] + 1
        cluster_pore_mask = np.isin(valid_labeled_pores, pore_label_indices)
        coords = np.column_stack(np.where(cluster_pore_mask))
        if len(coords) == 0:
            continue

        if len(coords) > 30000:
            np.random.seed(42)
            idx = np.random.choice(len(coords), 30000, replace=False)
            coords = coords[idx]

        scatters.append(
            go.Scatter3d(
                x=coords[:, 1], y=coords[:, 2], z=coords[:, 0],
                mode='markers',
                marker=dict(size=size, color=color, opacity=opacity),
                name=name
            )
        )

    fig = go.Figure(data=scatters)
    fig.update_layout(
        title=(f"Shell-Aware Graph Clustering ({num_chambers} Chambers) | "
               f"Otsu: {size_threshold:.1f} | k={k} | Pruned: {edges_pruned}/{edges_total}"),
        scene=dict(
            xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
            aspectmode='data'
        ),
        paper_bgcolor='black', plot_bgcolor='black', font=dict(color='white')
    )

    fig.write_html(output_html)
    print(f"Saved: {output_html}")
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Shell-Aware Graph Clustering for Pore-Chamber Assignment"
    )
    parser.add_argument("input", type=str, help="Path to input cleaned .npy volume")
    parser.add_argument("output", type=str, help="Path to output HTML visualizer")
    parser.add_argument("--k", type=int, default=12,
                        help="Number of KNN neighbors (default: 12)")
    parser.add_argument("--line-samples", type=int, default=25,
                        help="Points sampled per edge for shell crossing check (default: 25)")
    parser.add_argument("--min-chamber", type=int, default=5,
                        help="Minimum pores to count as a chamber (default: 5)")

    args = parser.parse_args()
    process_volume(args.input, args.output,
                   k_neighbors=args.k,
                   n_line_samples=args.line_samples,
                   min_chamber_pores=args.min_chamber)
