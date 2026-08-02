import argparse
import os
import numpy as np
import scipy.ndimage as ndimage

def remove_outliers_lcc(pred_path, out_path, warn_threshold):
    print(f"Loading prediction: {pred_path}")
    pred_vol = np.load(pred_path)
    
    # Background is 0. Shell (Chamber=1, Pores=2) is > 0
    shell_mask = (pred_vol > 0)
    
    print("Running 3D Connected Components analysis...")
    # Find all contiguous structures (blobs) in the 3D volume
    labeled_array, num_features = ndimage.label(shell_mask)
    
    print(f"Found {num_features} distinct structures.")
    
    if num_features > 1:
        # Calculate the volume (voxel count) of each structure
        sizes = np.bincount(labeled_array.ravel())
        
        # sizes[0] is the background, ignore it
        sizes[0] = 0
        
        # The true shell should be the absolute largest contiguous mass
        largest_cc_label = sizes.argmax()
        largest_cc_size = sizes[largest_cc_label]
        
        print(f"Largest structure has {largest_cc_size} voxels.")
        
        # Create a boolean mask of ONLY the main shell
        main_body_mask = (labeled_array == largest_cc_label)
        
        # Calculate how much flying debris we are about to erase
        total_shell_voxels = shell_mask.sum()
        outliers_removed = total_shell_voxels - largest_cc_size
        percent_removed = (outliers_removed / total_shell_voxels) * 100
        
        print(f"Erasing {outliers_removed} totally disconnected floating outlier voxels ({percent_removed:.3f}% of predicted mass).")
        
        if percent_removed > warn_threshold:
            print(f"\n=======================================================")
            print(f"⚠️  WARNING: MASSIVE OUTLIER REMOVAL DETECTED!")
            print(f"Algorithm is attempting to erase {percent_removed:.2f}% (> {warn_threshold}%) of the total shell mass.")
            print(f"This implies the neural network predicted a huge mass that is completely disconnected from the main body.")
            print(f"Check the visualizer for this volume to ensure the shell wasn't falsely split in half.")
            print(f"=======================================================\n")
        
        # Zero out anything that isn't connected to the main body
        # ~main_body_mask means "NOT main body"
        pred_vol[~main_body_mask] = 0
    else:
        print("Volume is already perfectly contiguous. No outliers found.")
        
    print(f"Saving cleaned volume to: {out_path}")
    # Ensure directory exists
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    np.save(out_path, pred_vol)
    print("Done.\n")

def main():
    parser = argparse.ArgumentParser(description="Remove floating 3D outliers using LCC")
    parser.add_argument("--pred", type=str, required=True, help="Input predicted .npy file")
    parser.add_argument("--out", type=str, required=True, help="Output cleaned .npy file")
    parser.add_argument("--warn-threshold", type=float, default=0.5, help="Percentage of mass removal to trigger a critical warning (default: 0.5%)")
    args = parser.parse_args()
    
    remove_outliers_lcc(args.pred, args.out, args.warn_threshold)

if __name__ == "__main__":
    main()
