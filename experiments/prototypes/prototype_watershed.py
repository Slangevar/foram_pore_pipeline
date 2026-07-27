import numpy as np
import scipy.ndimage as ndimage
from skimage.filters import threshold_otsu
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
import plotly.graph_objects as go
import time
import argparse
import sys

def process_volume(volume_path, output_html, downsample_points=150000):
    print(f"Loading {volume_path}...")
    try:
        data = np.load(volume_path)
    except Exception as e:
        print(f"Error loading {volume_path}: {e}")
        sys.exit(1)

    # Validate dims
    if data.ndim == 4:
        # Expected from previous network output [1, X, Y, Z]
        data = data[0]
        
    print(f"Volume Shape: {data.shape}")
    
    # 1. Extract Masks
    # Assume 1 = Shell/Chamber, 2 = Pores (based on previous predict pipeline)
    print("Extracting class coordinates...")
    shell_mask = (data == 1)
    pore_mask = (data == 2)
    
    # 2. Pore Noise Removal (Otsu Thresholding)
    print("Running 3D Connected Components on Pores...")
    labeled_pores, num_pores = ndimage.label(pore_mask)
    if num_pores == 0:
        print("No pores found in this volume!")
        sys.exit(0)
        
    print(f"Found {num_pores} raw pores.")
    pore_volumes = ndimage.sum(pore_mask, labeled_pores, range(1, num_pores + 1))
    
    # Filter out anything less than 1 voxel to avoid log(0) and extreme noise
    valid_volumes = pore_volumes[pore_volumes > 0]
    
    if len(valid_volumes) > 0:
        log_vols = np.log(valid_volumes)
        # Bimodal Otsu Split
        try:
            log_thresh = threshold_otsu(log_vols)
            size_threshold = np.exp(log_thresh)
            print(f"Otsu log threshold: {log_thresh:.3f} -> Min Pore Volume Voxel Count: {size_threshold:.1f}")
            
            # Keep only pores larger than the mathematical bimodal split
            pores_to_keep_labels = np.where(pore_volumes >= size_threshold)[0] + 1
            filtered_pore_mask = np.isin(labeled_pores, pores_to_keep_labels)
            
            removed_count = num_pores - len(pores_to_keep_labels)
            print(f"Removed {removed_count} tiny dust pores. Kept {len(pores_to_keep_labels)} valid structural pores.")
        except Exception as e:
            print(f"Otsu thresholding failed (likely monomodal noise): {e}. Keeping all valid pores.")
            filtered_pore_mask = pore_mask
    else:
        filtered_pore_mask = pore_mask
    
    # 3. 3D Bounding Constraints
    # Dilation of the shell mask by 3 voxels to "absorb" surface pores
    print("Dilating shell wall by 3 voxels to validate pore contact...")
    dilated_shell = ndimage.binary_dilation(shell_mask, iterations=3)
    
    # Mathematical AND check. Only keep pores that physically intersect the dilated shell
    final_pore_mask = filtered_pore_mask & dilated_shell
    
    # Check what was removed
    valid_labeled_pores, n_valid = ndimage.label(final_pore_mask)
    print(f"Spatial Validation complete: {len(np.unique(labeled_pores[filtered_pore_mask & ~dilated_shell])) - 1} floating pores destroyed.")
    print(f"Final Count: {n_valid} legitimate touching pores.")


    # 4. 3D Distance Transform Watershed Segmentation of the Shell
    print("Calculating 3D Distance Transform of the main Shell mass...")
    t0 = time.time()
    dist_transform = ndimage.distance_transform_edt(shell_mask)
    
    print(f"Finding local maxima (Chamber Centers)...")
    # Using a 3D structural footprint to find peaks.
    # min_distance ensures we don't over-segment tiny bumps inside a single chamber
    local_maxi = peak_local_max(dist_transform, min_distance=15, labels=shell_mask)
    
    # Create marker mask for the peaks
    markers = np.zeros(dist_transform.shape, dtype=bool)
    markers[tuple(local_maxi.T)] = True
    
    print(f"Found {len(local_maxi)} distinct Chamber centers.")
    
    markers_labeled, num_chambers = ndimage.label(markers)
    print(f"Running 3D Watershed (Segmenting {num_chambers} Chambers)...")
    
    # Invert distance transform because watershed climbs up from low points
    chamber_labels = watershed(-dist_transform, markers_labeled, mask=shell_mask)
    print(f"Watershed complete in {time.time()-t0:.2f}s.")
    
    
    # 5. Visualizer Prep: Downsampling coordinates for HTML
    print("\nPreparing Visualizer Data...")
    
    scatters = []
    
    # Color palette for distinct chambers
    import plotly.express as px
    colors = px.colors.qualitative.Alphabet + px.colors.qualitative.Dark24
    
    # Add each physical Watershed Chamber lobe
    for i in range(1, num_chambers + 1):
        c_mask = (chamber_labels == i)
        coords = np.column_stack(np.where(c_mask))
        if len(coords) == 0: continue
            
        if len(coords) > downsample_points:
            np.random.seed(42)
            idx = np.random.choice(len(coords), downsample_points, replace=False)
            coords = coords[idx]
            
        color = colors[i % len(colors)]
        
        scatters.append(
            go.Scatter3d(
                x=coords[:, 1], y=coords[:, 2], z=coords[:, 0], # ZYX to XYZ logic for visual
                mode='markers',
                marker=dict(size=1.5, color=color, opacity=0.15),
                name=f"Chamber Lobe {i}"
            )
        )
        
    # Color the validated pores belonging to those chambers
    # We figure out which chamber a pore belongs to simply by checking the dilated shell overlapping label at that coordinate
    print("Mapping Validated Pores to Parent Chambers...")
    
    for i in range(1, num_chambers + 1):
        # The true physical boundary of the watershed specific chamber
        c_mask = (chamber_labels == i)
        # Dilate just this specific chamber to capture the surface pores touching it
        dilated_c_mask = ndimage.binary_dilation(c_mask, iterations=3)
        
        # Look for the validated pores that intersect this specific chamber
        chamber_pores = final_pore_mask & dilated_c_mask
        
        coords = np.column_stack(np.where(chamber_pores))
        if len(coords) == 0: continue
            
        if len(coords) > downsample_points:
            np.random.seed(42)
            idx = np.random.choice(len(coords), downsample_points, replace=False)
            coords = coords[idx]
            
        color = colors[i % len(colors)]
        
        scatters.append(
            go.Scatter3d(
                x=coords[:, 1], y=coords[:, 2], z=coords[:, 0],
                mode='markers',
                marker=dict(size=3, color=color, opacity=1.0, symbol='diamond'),
                name=f"Pores (Chamber {i})"
            )
        )

    print(f"Building Plotly 3D scatter plot: {output_html} ...")
    fig = go.Figure(data=scatters)
    fig.update_layout(
         title=f"3D Watershed Segmentation Prototype ({volume_path})",
         scene=dict(
             xaxis=dict(title='Y', visible=False),
             yaxis=dict(title='X', visible=False),
             zaxis=dict(title='Z', visible=False),
             aspectmode='data'
         ),
         paper_bgcolor='rgba(0,0,0,1)',
         plot_bgcolor='rgba(0,0,0,1)',
         font=dict(color='white')
    )
    
    fig.write_html(output_html)
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prototype 3D Watershed Chamber Segmentation")
    parser.add_argument("input", type=str, help="Path to input cleaned numpy volume")
    parser.add_argument("output", type=str, help="Path to output HTML visualizer")
    
    args = parser.parse_args()
    process_volume(args.input, args.output)
