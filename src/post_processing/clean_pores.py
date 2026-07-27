import numpy as np
import scipy.ndimage as ndimage
from skimage.filters import threshold_otsu
import argparse
import sys
import os

def clean_pores(input_path, output_path):
    print(f"Loading {input_path}...")
    try:
        data = np.load(input_path)
    except Exception as e:
        print(f"Error loading {input_path}: {e}")
        sys.exit(1)

    if data.ndim == 4:
        # Expected from previous network output [1, X, Y, Z]
        out_data = data.copy()
        vol = data[0]
    else:
        out_data = data.copy()
        vol = data

    print(f"Volume Shape: {vol.shape}")
    
    shell_mask = (vol == 1)
    pore_mask = (vol == 2)
    
    # ── 1. Pore Noise Removal (Otsu Thresholding) ──
    print("\n--- Extracting valid pores (Otsu) ---")
    structure_6conn = np.array([
        [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
        [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
        [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
    ])
    labeled_pores, num_pores = ndimage.label(pore_mask, structure=structure_6conn)
    
    if num_pores == 0:
        print("No pores found. Saving original array.")
        np.save(output_path, out_data)
        return

    pore_volumes = ndimage.sum(pore_mask, labeled_pores, range(1, num_pores + 1))
    valid_volumes = pore_volumes[pore_volumes > 0]
    
    if len(valid_volumes) > 0:
        log_thresh = threshold_otsu(np.log(valid_volumes))
        size_threshold = np.exp(log_thresh)
        pores_to_keep_labels = np.where(pore_volumes >= size_threshold)[0] + 1
        print(f"Otsu log threshold kept {len(pores_to_keep_labels)} pores (>{size_threshold:.1f} voxels).")
    else:
        pores_to_keep_labels = np.arange(1, num_pores + 1)
        print("Could not calculate Otsu threshold. Keeping all pores for spatial validation.")

    # ── 2. Spatial Validation (Must touch shell) ──
    print("\n--- Spatial Validation ---")
    dilated_shell = ndimage.binary_dilation(shell_mask, iterations=3)
    filtered_pore_mask = np.isin(labeled_pores, pores_to_keep_labels) & dilated_shell
    
    valid_labeled_pores, n_valid = ndimage.label(filtered_pore_mask, structure=structure_6conn)
    print(f"Final Count: {n_valid} spatially valid pores touching the shell.")
    
    # ── 3. Apply changes and save ──
    # Zero out all original pores
    if out_data.ndim == 4:
        out_data[0][out_data[0] == 2] = 0
        # Restore only the valid pores
        out_data[0][filtered_pore_mask] = 2
    else:
        out_data[out_data == 2] = 0
        out_data[filtered_pore_mask] = 2
        
    print(f"Saving cleaned array to: {output_path}")
    np.save(output_path, out_data)
    print("Clean successfully completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Isolate and clean mathematical pore noise.")
    parser.add_argument("input", type=str, help="Path to input .npy volume")
    parser.add_argument("output", type=str, help="Path to save cleaned .npy volume")
    
    args = parser.parse_args()
    clean_pores(args.input, args.output)
