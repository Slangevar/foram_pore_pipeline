import os
import sys
import numpy as np
import torch
from skimage import io
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.append(project_root)

from src import model, predict

def main():
    # Paths
    val_image_dir = os.path.join(project_root, "data", "val", "images")
    image_files = sorted([f for f in os.listdir(val_image_dir) if f.lower().endswith('.tiff')])
    if not image_files:
        raise FileNotFoundError(f"No .tiff files found in {val_image_dir}")
    
    # Use the second image as requested by user edits
    img_path = os.path.join(val_image_dir, image_files[1])
    print(f"Loading validation image: {img_path}")
    img = io.imread(img_path).astype(np.float32) / 255.0
    
    # Load model checkpoint
    ckpt_path = os.path.join(project_root, "model", "model.ckpt")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Loading model from {ckpt_path} on device {device}")
    
    # Loading model to pass to predict_single_slice
    loaded_model = model.SegmentationModel.load_from_checkpoint(checkpoint_path=ckpt_path)
    loaded_model.to(device)
    loaded_model.eval()
    
    # Use the central function from src/predict.py
    print("Running prediction using predict_single_slice from src/predict.py...")
    y_pred = predict.predict_single_slice(
        image_slice=img,
        model_instance=loaded_model,
        device=device
    )
    
    # Save prediction as TIFF
    pred_path = os.path.join(project_root, "val_slice_pred.tiff")
    io.imsave(pred_path, y_pred.astype(np.uint8))
    print(f"Saved prediction TIFF to {pred_path}")
    
    # Visualization: side-by-side original & prediction
    # Custom colormap: Red, Yellow, Green
    custom_cmap = ListedColormap(['red', 'yellow', 'green'])
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # Handle image display (squeeze if needed)
    display_img = img.squeeze()
    if display_img.ndim > 2 and display_img.shape[0] == 1:
        display_img = display_img[0]
        
    axes[0].imshow(display_img, cmap='gray')
    axes[0].set_title(f'Original: {os.path.basename(img_path)}')
    axes[0].axis('off')
    
    im = axes[1].imshow(y_pred, cmap=custom_cmap, vmin=0, vmax=2)
    axes[1].set_title('Prediction (Red/Yellow/Green)')
    axes[1].axis('off')
    
    # Add colorbar to show mapping
    cbar = fig.colorbar(im, ax=axes[1], ticks=[0, 1, 2], fraction=0.046, pad=0.04)
    cbar.ax.set_yticklabels(['Class 0', 'Class 1', 'Class 2'])
    
    vis_path = os.path.join(project_root, "val_slice_vis.png")
    plt.tight_layout()
    plt.savefig(vis_path, dpi=150)
    print(f"Saved visualization PNG to {vis_path}")
    print(f"Unique classes in prediction: {np.unique(y_pred)}")

if __name__ == "__main__":
    main()
