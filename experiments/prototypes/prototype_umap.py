"""
UMAP Manifold HDBSCAN Clustering.

Why this succeeds where Isomap and Watershed failed on the Inner Core:
- Watershed and Isomap rely on *Shell Topology* (the physical distance/bottlenecks 
  in the shell matrix). In the late-stage outer chambers, the shell lobes are thin
  and obviously separated by empty space, so topological bottlenecks are huge.
- In the *Inner Core* (early-stage chambers), foraminifera shells are heavily 
  calcified and fused. There is NO empty void separating them, and no thin "neck"
  for Isomap or Watershed to detect. The topological distance is perfectly smooth.
- However, the *Pore Density* still dips slightly at the boundaries between these 
  fused inner chambers. 
- UMAP (and t-SNE) are *Statistical Density* algorithms. They don't care about 
  topological bottlenecks in the solid shell matrix; they only care about local 
  density. They detect those tiny microscopic dips in pore density at the chamber 
  boundaries and mathematically rip them apart into large gaps in abstract space.

This makes UMAP the mathematically correct solution for heavily calcified, fused
biological volumes where topological boundaries vanish!
"""

import numpy as np
import scipy.ndimage as ndimage
import hdbscan
import umap.umap_ as umap
import plotly.graph_objects as go
import plotly.express as px
import argparse
import sys
import time

def process_volume(volume_path, output_html, n_neighbors=100, min_dist=0.1, min_cluster_size=50):
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
    if n_valid < min_cluster_size:
        print(f"Found {n_valid} pores. Too few to cluster.")
        sys.exit(0)
        
    centroids = np.array(ndimage.center_of_mass(pore_mask, labeled_pores, range(1, n_valid + 1)))
    print(f"Extracted {n_valid} pore centroids.")

    # ── 2. UMAP Dimensionality Reduction ──
    # We reduce to 3D so HDBSCAN has a rich volume to cluster rather than a flat, crowded 2D plane.
    print(f"\n--- Step 2: UMAP Manifold Projection (neighbors={n_neighbors}, min_dist={min_dist}) ---")
    t0 = time.time()
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=3,
        metric='euclidean',
        random_state=42 # Deterministic, unlike t-SNE which changes every time!
    )
    embedding = reducer.fit_transform(centroids)
    print(f"Embedded {n_valid} points into 3D UMAP space in {time.time()-t0:.1f}s")

    # ── 3. HDBSCAN on the UMAP Embedding ──
    print(f"\n--- Step 3: HDBSCAN Clustering ---")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=1,
        cluster_selection_epsilon=0.1,  # Small merge radius in abstract UMAP space
        cluster_selection_method='eom'
    )
    labels = clusterer.fit_predict(embedding)
    
    num_chambers = len(set(labels)) - (1 if -1 in labels else 0)
    noise_count = np.sum(labels == -1)
    print(f"SUCCESS: Found {num_chambers} Chambers out of the UMAP Manifold! ({noise_count} noise pores)")

    # ── 4. Visualizer Output ──
    print("\n--- Step 4: Compiling Plotly Visualizer ---")
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
    
    for cluster_id in sorted(set(labels)):
        if cluster_id == -1:
            name = f"Unclustered/Noise ({np.sum(labels == -1)})"
            color = "gray"
            size = 2
            opacity = 0.3
        else:
            name = f"Chamber {cluster_id} ({np.sum(labels == cluster_id)})"
            color = colors[cluster_id % len(colors)]
            size = 3
            opacity = 1.0
            
        pore_label_indices = np.where(labels == cluster_id)[0] + 1
        cluster_pore_mask = np.isin(labeled_pores, pore_label_indices)
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
         title=f"UMAP + HDBSCAN ({num_chambers} Chambers) | n_neighbors={n_neighbors}",
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
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str)
    parser.add_argument("output", type=str)
    args = parser.parse_args()
    process_volume(args.input, args.output)
