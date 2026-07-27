"""
Geodesic HDBSCAN (Isomap) on Foraminifera Pore Manifold.

Algorithm:
1. Load a pre-cleaned numpy volume (artifacts already removed by clean_pores.py).
2. Build a Local K-Nearest-Neighbor (KNN) graph of pore centroids.
3. Edge Pruning (The "Manifold" constraint):
   If the straight Euclidean line between two pores crosses the *Empty Background Void*
   (meaning it cuts through the hollow chamber core instead of curving along the
   physical shell wall), we delete that edge!
4. Calculate All-Pairs Shortest Path (APSP) using Dijkstra on the surviving edges.
   This creates a custom Geodesic Distance Matrix where distance = "travel distance 
   bending along the curves of the shell wall".
5. Feed the Geodesic Distance Matrix into HDBSCAN using `metric='precomputed'`.
6. Output biological separation visualizations to HTML.
"""

import numpy as np
import scipy.ndimage as ndimage
from scipy.spatial import cKDTree
import scipy.sparse as sparse
from scipy.sparse.csgraph import shortest_path
import hdbscan
import plotly.graph_objects as go
import plotly.express as px
import argparse
import sys
import time

def line_crosses_void(p1, p2, background_mask, n_samples=30):
    shape = np.array(background_mask.shape)
    for t in np.linspace(0, 1, n_samples):
        point = (1 - t) * p1 + t * p2
        idx = np.round(point).astype(int)
        if np.any(idx < 0) or np.any(idx >= shape):
            continue
        # If the pixel along the line is true (which means it's Background/Void)
        if background_mask[idx[0], idx[1], idx[2]]:
            return True
    return False

def process_volume(volume_path, output_html, k_neighbors=30, min_cluster_size=50):
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
    
    # ── 1. Verify Clean Pores and Extract Centroids ──
    structure_6conn = np.array([
        [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
        [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
        [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
    ])
    
    labeled_pores, n_valid = ndimage.label(pore_mask, structure=structure_6conn)
    if n_valid == 0:
        print("No pores found in this volume!")
        sys.exit(0)
    elif n_valid < min_cluster_size:
        print(f"Found only {n_valid} pores, too few to cluster.")
        sys.exit(0)
        
    print(f"Found {n_valid} pre-validated pores. Extracting centroids...")
    centroids = np.array(ndimage.center_of_mass(pore_mask, labeled_pores, range(1, n_valid + 1)))

    # Define the "Manifold" (The physical tissue where paths are allowed to travel)
    # We dilate the tissue slightly to allow paths that graze the bumpy surface.
    tissue_mask = shell_mask | pore_mask
    dilated_tissue = ndimage.binary_dilation(tissue_mask, iterations=2)
    background_mask = ~dilated_tissue  # The "Empty Void" we cannot cross
    
    # ── 2. Build Euclidean KNN ──
    print(f"\n--- Step 2: Building Euclidean KNN Graph (k={k_neighbors}) ---")
    k = min(k_neighbors, n_valid - 1)
    tree = cKDTree(centroids)
    distances, indices = tree.query(centroids, k=k + 1)

    # ── 3. Graph Pruning (Void Penetration) ──
    print("\n--- Step 3: Void Penetration Pruning ---")
    t0 = time.time()
    
    row = []
    col = []
    data_weights = []
    
    edges_total = 0
    edges_pruned = 0
    
    for i in range(n_valid):
        for j_idx in range(1, k + 1):
            j = indices[i, j_idx]
            if j <= i: continue
            
            edges_total += 1
            # If the shortcut cuts through the empty chamber, DELETE the edge!
            if line_crosses_void(centroids[i], centroids[j], background_mask, n_samples=30):
                edges_pruned += 1
            else:
                weight = distances[i, j_idx]
                row.extend([i, j])
                col.extend([j, i])
                data_weights.extend([weight, weight])
                
    print(f"Checked {edges_total} local edges. Pruned {edges_pruned} shortcuts leaping across biological voids.")
    print(f"Time: {time.time()-t0:.1f}s")
    
    if len(row) == 0:
        print("All edges pruned! Pores are completely disconnected from one another across the manifold. Exiting.")
        sys.exit(0)
        
    adj_matrix = sparse.csr_matrix((data_weights, (row, col)), shape=(n_valid, n_valid))

    # ── 4. Calculate Geodesic Distances (Dijkstra) ──
    print("\n--- Step 4: Computing Geodesic Surface Distances ---")
    t0 = time.time()
    geodesic_dist_matrix = shortest_path(csgraph=adj_matrix, directed=False)
    
    finite_mask = ~np.isinf(geodesic_dist_matrix)
    max_dist = np.max(geodesic_dist_matrix[finite_mask]) if finite_mask.any() else 100.0
    geodesic_dist_matrix[~finite_mask] = max_dist * 5.0
    np.fill_diagonal(geodesic_dist_matrix, 0)
    print(f"Calculated {n_valid}x{n_valid} Distance Matrix in {time.time()-t0:.1f}s")

    # ── 5. Isomap-HDBSCAN Clustering ──
    print(f"\n--- Step 5: Clustering via Geodesic HDBSCAN ---")
    clusterer = hdbscan.HDBSCAN(
        metric='precomputed',
        min_cluster_size=min_cluster_size,
        min_samples=3,
        cluster_selection_method='eom',
        cluster_selection_epsilon=10.0,
    )
    labels = clusterer.fit_predict(geodesic_dist_matrix)
    
    num_chambers = len(set(labels)) - (1 if -1 in labels else 0)
    noise_count = np.sum(labels == -1)
    print(f"SUCCESS: Found {num_chambers} Manifold Chambers! ({noise_count} pores marked as noise)")

    # ── 6. Visualizer Output ──
    print("\n--- Step 6: Compiling Plotly HDBSCAN Visualizer ---")
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
         title=f"Geodesic Manifold HDBSCAN ({num_chambers} Chambers)",
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
