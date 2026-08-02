"""
t-SNE + HDBSCAN Clustering (Battle-Tested Parameters from post_processing.py).

This script ports the user's proven t-SNE + HDBSCAN pipeline into the new
modular architecture, reading from pre-cleaned volumes in data/pores_cleaned/.

Additional feature: Lonely pores (HDBSCAN noise, label == -1) are automatically
merged into their nearest cluster based on centroid proximity in t-SNE space.
"""

import numpy as np
import scipy.ndimage as ndimage
from scipy.spatial import cKDTree
from sklearn.manifold import TSNE
import hdbscan
import plotly.graph_objects as go
import plotly.express as px
import argparse
import sys
import time
import os

def process_volume(volume_path, output_html,
                   perplexity=20, angle=0.2,
                   min_cluster_size=50, cluster_selection_epsilon=0.5,
                   min_samples=1):
    print(f"Loading pre-cleaned volume: {volume_path}...")
    try:
        data = np.load(volume_path)
    except Exception as e:
        print(f"Error loading {volume_path}: {e}")
        sys.exit(1)

    if data.ndim == 4:
        data = data[0]

    shell_mask = (data == 1)
    pore_mask = (data == 2)

    # ── 1. Extract Valid Centroids ──
    print("\n--- Step 1: Extracting Centroids ---")
    structure_6conn = np.array([
        [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
        [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
        [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
    ])

    labeled_pores, n_valid = ndimage.label(pore_mask, structure=structure_6conn)
    if n_valid == 0:
        print("Found 0 pores. Nothing to cluster.")
        sys.exit(0)

    centroids_3d = np.array(ndimage.center_of_mass(
        pore_mask, labeled_pores, range(1, n_valid + 1)
    ))
    print(f"Extracted {n_valid} pore centroids.")

    # Fallback path: if the volume has too few pores for stable HDBSCAN,
    # keep it editable by assigning all pores to one chamber.
    if n_valid < min_cluster_size:
        print(f"Found {n_valid} pores. Too few for HDBSCAN; using single-cluster fallback.")
        labels = np.full(n_valid, 2, dtype=np.int32)
        embedding = centroids_3d[:, :2].astype(np.float64)
        if embedding.shape[1] < 2:
            embedding = np.pad(embedding, ((0, 0), (0, 2 - embedding.shape[1])), mode='constant')

        final_chambers = 1
        print("Fallback Result: 1 chamber with all pores assigned.")

        results_dir = os.path.join(os.path.dirname(output_html), '..', 'clustering_results')
        os.makedirs(results_dir, exist_ok=True)
        vol_basename = os.path.basename(volume_path).replace('_pores_cleaned.npy', '')
        results_path = os.path.join(results_dir, f"{vol_basename}_clustering_results.npz")

        np.savez_compressed(
            results_path,
            centroids_3d=centroids_3d,
            embedding_2d=embedding,
            labels=labels,
            volume_path=volume_path
        )
        print(f"Saved clustering results: {results_path}")

        print("\n--- Step 6: Compiling Plotly Visualizer ---")
        colors = px.colors.qualitative.Alphabet + px.colors.qualitative.Dark24
        scatters = []

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

        color = colors[2 % len(colors)]
        scatters.append(
            go.Scatter3d(
                x=centroids_3d[:, 1], y=centroids_3d[:, 2], z=centroids_3d[:, 0],
                mode='markers',
                marker=dict(size=4, color=color, opacity=1.0),
                name=f"Chamber 2 ({n_valid} pores)"
            )
        )

        fig = go.Figure(data=scatters)
        fig.update_layout(
            title=(f"t-SNE + HDBSCAN Fallback ({final_chambers} Chamber) | "
                   f"n_pores={n_valid}"),
            scene=dict(
                xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
                aspectmode='data'
            ),
            legend=dict(itemsizing='constant'),
            paper_bgcolor='black', plot_bgcolor='black', font=dict(color='white')
        )

        fig.write_html(output_html)
        print(f"Saved: {output_html}")
        print("Done!")
        return

    # ── 2. t-SNE Dimensionality Reduction (Original Parameters) ──
    print(f"\n--- Step 2: t-SNE Projection (perplexity={perplexity}, angle={angle}) ---")
    t0 = time.time()
    tsne = TSNE(n_components=2, perplexity=perplexity, angle=angle, random_state=42)
    embedding = tsne.fit_transform(centroids_3d)
    print(f"Embedded {n_valid} points into 2D t-SNE space in {time.time()-t0:.1f}s")

    # ── 3. HDBSCAN on the t-SNE Embedding (Original Parameters) ──
    print(f"\n--- Step 3: HDBSCAN Clustering ---")
    print(f"  min_cluster_size={min_cluster_size}, "
          f"cluster_selection_epsilon={cluster_selection_epsilon}, "
          f"min_samples={min_samples}")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        cluster_selection_epsilon=cluster_selection_epsilon,
        min_samples=min_samples,
        metric='euclidean',
        cluster_selection_method='eom'
    )
    labels = clusterer.fit_predict(embedding) + 2  # Shift: noise becomes 1, clusters start at 2+

    num_chambers = len(set(labels)) - (1 if 1 in labels else 0)
    noise_count = np.sum(labels == 1)
    print(f"HDBSCAN found {num_chambers} chambers! ({noise_count} lonely/noise pores)")

    # ── 4. Merge Lonely Pores into Nearest Cluster ──
    if noise_count > 0:
        print(f"\n--- Step 4: Merging {noise_count} lonely pores into nearest chambers ---")
        noise_mask = (labels == 1)
        cluster_mask = (labels >= 2)

        if np.any(cluster_mask):
            # Build a KD-Tree of the t-SNE coordinates of clustered pores
            clustered_embedding = embedding[cluster_mask]
            clustered_labels = labels[cluster_mask]
            tree = cKDTree(clustered_embedding)

            # For each noise pore, find nearest clustered pore and adopt its label
            noise_embedding = embedding[noise_mask]
            _, nearest_indices = tree.query(noise_embedding, k=1)
            labels[noise_mask] = clustered_labels[nearest_indices]
            print(f"All {noise_count} lonely pores merged into their nearest chamber.")
        else:
            labels[:] = 2
            print("HDBSCAN produced only noise; applied single-cluster fallback.")
    else:
        print("\n--- Step 4: No lonely pores to merge. ---")

    # Final count
    final_chambers = len(set(labels))
    print(f"\nFinal Result: {final_chambers} chambers with 0 noise pores.")

    # ── 5. Save Clustering Results (Science Output) ──
    results_dir = os.path.join(os.path.dirname(output_html), '..', 'clustering_results')
    os.makedirs(results_dir, exist_ok=True)
    vol_basename = os.path.basename(volume_path).replace('_pores_cleaned.npy', '')
    results_path = os.path.join(results_dir, f"{vol_basename}_clustering_results.npz")
    
    np.savez_compressed(
        results_path,
        centroids_3d=centroids_3d,
        embedding_2d=embedding,
        labels=labels,
        volume_path=volume_path
    )
    print(f"Saved clustering results: {results_path}")

    # ── 6. Visualization ──
    print("\n--- Step 6: Compiling Plotly Visualizer ---")
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

    # Colored centroids (lighter weight than voxels for the static HTML)
    for cluster_id in sorted(set(labels)):
        mask = (labels == cluster_id)
        count = np.sum(mask)
        name = f"Chamber {cluster_id} ({count} pores)"
        color = colors[cluster_id % len(colors)]
        
        c_subset = centroids_3d[mask]
        
        scatters.append(
            go.Scatter3d(
                x=c_subset[:, 1], y=c_subset[:, 2], z=c_subset[:, 0],
                mode='markers',
                marker=dict(size=4, color=color, opacity=1.0),
                name=name
            )
        )

    fig = go.Figure(data=scatters)
    fig.update_layout(
        title=(f"t-SNE + HDBSCAN ({final_chambers} Chambers) | "
               f"perplexity={perplexity}, ε={cluster_selection_epsilon}"),
        scene=dict(
            xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
            aspectmode='data'
        ),
        legend=dict(itemsizing='constant'),
        paper_bgcolor='black', plot_bgcolor='black', font=dict(color='white')
    )

    fig.write_html(output_html)
    print(f"Saved: {output_html}")
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="t-SNE + HDBSCAN Clustering (Original Parameters)"
    )
    parser.add_argument("input", type=str, help="Path to pre-cleaned .npy volume")
    parser.add_argument("output", type=str, help="Path to output HTML visualizer")
    args = parser.parse_args()
    process_volume(args.input, args.output)
