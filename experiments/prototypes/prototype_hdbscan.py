import numpy as np
import scipy.ndimage as ndimage
from skimage.filters import threshold_otsu
import hdbscan
import plotly.graph_objects as go
import plotly.express as px
import argparse
import sys

def process_volume(volume_path, output_html, min_cluster_size=15):
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
    print("Extracting valid pores...")
    structure_6conn = np.array([
        [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
        [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
        [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
    ])
    labeled_pores, num_pores = ndimage.label(pore_mask, structure=structure_6conn)
    if num_pores == 0:
        print("No pores found in this volume!")
        sys.exit(0)
        
    pore_volumes = ndimage.sum(pore_mask, labeled_pores, range(1, num_pores + 1))
    valid_volumes = pore_volumes[pore_volumes > 0]
    
    size_threshold = 0
    if len(valid_volumes) > 0:
        log_vols = np.log(valid_volumes)
        try:
            log_thresh = threshold_otsu(log_vols)
            size_threshold = np.exp(log_thresh)
            pores_to_keep_labels = np.where(pore_volumes >= size_threshold)[0] + 1
            print(f"Otsu threshold: {size_threshold:.1f} voxels. Kept {len(pores_to_keep_labels)} / {num_pores} pores.")
        except:
            pores_to_keep_labels = np.arange(1, num_pores + 1)
    else:
        pores_to_keep_labels = np.arange(1, num_pores + 1)

    # Spatial validation: pores must touch the dilated shell
    dilated_shell = ndimage.binary_dilation(shell_mask, iterations=3)
    filtered_pore_mask = np.isin(labeled_pores, pores_to_keep_labels) & dilated_shell
    
    # Re-label the filtered pores
    valid_labeled_pores, n_valid = ndimage.label(filtered_pore_mask, structure=structure_6conn)
    print(f"Spatially valid pores: {n_valid}")
    
    if n_valid < min_cluster_size:
        print("Not enough pores to cluster.")
        sys.exit(0)

    # ── 2. Extract Centroids & Volumes ──
    centroids = np.array(ndimage.center_of_mass(filtered_pore_mask, valid_labeled_pores, range(1, n_valid + 1)))
    volumes = ndimage.sum(filtered_pore_mask, valid_labeled_pores, range(1, n_valid + 1))
    volumes = np.array(volumes)

    # ── 3. Volume-Weighted Centroid Replication ──
    # The idea: replicate each centroid proportional to log(volume), so that
    # large structural pores contribute more density signal to HDBSCAN, making
    # the density cores of each chamber much sharper and the inter-chamber
    # voids more pronounced.
    log_weights = np.log1p(volumes)  # log(1 + volume) to avoid log(0)
    # Normalize to get integer replication counts (min 1, max ~10)
    max_reps = 10
    normalized_weights = (log_weights / log_weights.max()) * max_reps
    rep_counts = np.clip(np.round(normalized_weights).astype(int), 1, max_reps)
    
    # Build the weighted centroid array
    weighted_centroids = np.repeat(centroids, rep_counts, axis=0)
    # Track which original pore index each replicated row belongs to
    pore_indices_map = np.repeat(np.arange(n_valid), rep_counts)
    
    print(f"Volume-weighted replication: {len(centroids)} centroids -> {len(weighted_centroids)} weighted points")
    
    # ── 4. Native 3D HDBSCAN ──
    print(f"Running Volume-Weighted 3D HDBSCAN (min_cluster_size={min_cluster_size})...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=3,
        metric='euclidean',
        cluster_selection_method='eom',
    )
    weighted_labels = clusterer.fit_predict(weighted_centroids)
    
    # Map weighted labels back to original pore centroids via majority vote
    labels = np.full(n_valid, -1, dtype=int)
    for i in range(n_valid):
        mask = pore_indices_map == i
        pore_weighted_labels = weighted_labels[mask]
        # Take most common non-noise label, fallback to -1
        unique, counts = np.unique(pore_weighted_labels, return_counts=True)
        valid_mask = unique >= 0
        if valid_mask.any():
            labels[i] = unique[valid_mask][np.argmax(counts[valid_mask])]
    
    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise_count = np.sum(labels == -1)
    print(f"HDBSCAN found {num_clusters} chambers! ({noise_count} unclustered noise pores)")
    
    # ── 5. Visualization ──
    print("\nBuilding HTML Visualizer...")
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
    
    # Colored pores per cluster
    for cluster_id in sorted(set(labels)):
        if cluster_id == -1:
            name = "Unclustered Noise"
            color = "gray"
            size = 2
            opacity = 0.4
        else:
            name = f"Chamber {cluster_id}"
            color = colors[cluster_id % len(colors)]
            size = 3
            opacity = 1.0
            
        pore_label_indices = np.where(labels == cluster_id)[0] + 1  # 1-indexed
        cluster_pore_mask = np.isin(valid_labeled_pores, pore_label_indices)
        coords = np.column_stack(np.where(cluster_pore_mask))
        if len(coords) == 0: continue
        
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
         title=f"Volume-Weighted 3D HDBSCAN ({num_clusters} Chambers) | Otsu Threshold: {size_threshold:.1f}",
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
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str)
    parser.add_argument("output", type=str)
    args = parser.parse_args()
    process_volume(args.input, args.output)
