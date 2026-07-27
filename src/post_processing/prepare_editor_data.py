"""
Editor Data Preparation Tool.

This script takes a segmented volume and clustering results, then extracts 
the downsampled voxels needed for the 3D Cluster Editor. 

Decouples "Heavy" voxel extraction from "Iterative" scientific clustering.
"""

import numpy as np
import scipy.ndimage as ndimage
import argparse
import os
import sys
from collections import defaultdict

def prepare_data(volume_path, clustering_path, output_dir, chamber_ratio=0.1, max_chamber_voxels=100000):
    volume_path = os.path.abspath(volume_path)
    clustering_path = os.path.abspath(clustering_path)
    output_dir = os.path.abspath(output_dir)
    print(f"--- Preparing Editor Data: {os.path.basename(volume_path)} ---")
    
    # 1. Load Data
    data = np.load(volume_path)
    if data.ndim == 4: data = data[0]
    
    cluster_data = np.load(clustering_path, allow_pickle=True)
    labels = cluster_data['labels']
    centroids_3d = cluster_data['centroids_3d']
    
    shell_mask = (data == 1)
    pore_mask = (data == 2)

    # 2. Extract Labeled Pores
    print("Labeling pores for voxel extraction...")
    structure_6conn = np.array([
        [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
        [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
        [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
    ])
    labeled_pores, n_valid = ndimage.label(pore_mask, structure=structure_6conn)
    
    # 3. Downsample Pore Voxels (Class 2)
    max_voxels_per_pore = 30
    print(f"Downsampling {n_valid} pores (max {max_voxels_per_pore} vox/pore)...")
    
    all_z, all_y, all_x = np.where(labeled_pores > 0)
    all_pids = labeled_pores[all_z, all_y, all_x]
    
    pore_groups = defaultdict(list)
    for i in range(len(all_pids)):
        pore_groups[all_pids[i]].append(i)

    pore_voxels_list = []
    pore_voxels_owner = []
    for pid in range(1, n_valid + 1):
        indices = pore_groups.get(pid, [])
        if not indices: continue
        
        coords = np.column_stack((all_z[indices], all_y[indices], all_x[indices]))
        if len(coords) <= max_voxels_per_pore:
            sample = coords
        else:
            rng = np.random.RandomState(pid)
            idx = rng.choice(len(coords), max_voxels_per_pore, replace=False)
            sample = coords[idx]
        
        pore_voxels_list.append(sample)
        pore_voxels_owner.extend([pid - 1] * len(sample))

    pore_voxels = np.vstack(pore_voxels_list)
    pore_voxels_owner = np.array(pore_voxels_owner, dtype=np.int32)

    # 4. Sample Chamber Body Voxels (Class 1) for Context
    print(f"Sampling chamber body voxels (ratio={chamber_ratio}, max={max_chamber_voxels})...")
    chamber_z, chamber_y, chamber_x = np.where(shell_mask)
    n_chamber = len(chamber_z)
    
    if n_chamber > 0:
        n_target = int(n_chamber * chamber_ratio)
        n_sample = min(n_target, max_chamber_voxels)
        # Ensure at least a few points if any exist
        n_sample = max(n_sample, min(100, n_chamber))
        
        print(f"  Selecting {n_sample} voxels from {n_chamber} total.")
        rng_body = np.random.RandomState(42)
        idx_body = rng_body.choice(n_chamber, n_sample, replace=False)
        chamber_voxels = np.column_stack((chamber_z[idx_body], chamber_y[idx_body], chamber_x[idx_body]))
    else:
        chamber_voxels = np.zeros((0, 3))

    # 5. Save Final NPZ
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.basename(volume_path).replace('_pores_cleaned.npy', '')
    state_path = os.path.join(output_dir, f"{base}_cluster_state.npz")
    
    np.savez_compressed(
        state_path,
        centroids_3d=centroids_3d,
        labels=labels,
        labeled_pores=labeled_pores,
        shell_mask=shell_mask,
        volume_path=volume_path,
        pore_voxels=pore_voxels,
        pore_voxels_owner=pore_voxels_owner,
        chamber_voxels=chamber_voxels
    )
    print(f"Successfully prepared editor state: {state_path}")
    return state_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Voxel Data for 3D Editor")
    parser.add_argument("volume", help="Path to cleaned input .npy volume")
    parser.add_argument("clustering", help="Path to .npz with centroids and labels")
    parser.add_argument("outdir", help="Directory to save the editor state")
    parser.add_argument("--ratio", type=float, default=0.1, help="Sampling ratio for chamber voxels (default: 0.1)")
    parser.add_argument("--max-chamber-voxels", type=int, default=100000, help="Max absolute voxels for chamber shell (default: 100000)")
    args = parser.parse_args()
    
    prepare_data(args.volume, args.clustering, args.outdir, 
                 chamber_ratio=args.ratio, 
                 max_chamber_voxels=args.max_chamber_voxels)
