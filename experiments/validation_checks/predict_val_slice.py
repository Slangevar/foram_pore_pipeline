import os
import sys
import numpy as np
from skimage import io

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.append(project_root)

from src import predict

def main():
    # Find a validation image
    val_image_dir = os.path.join(project_root, "data", "val", "images")
    image_files = [f for f in os.listdir(val_image_dir) if f.lower().endswith('.tiff')]
    if not image_files:
        raise FileNotFoundError(f"No .tiff files found in {val_image_dir}")
    image_path = os.path.join(val_image_dir, image_files[0])
    print(f"Loading validation image: {image_path}")
    img = io.imread(image_path).astype(np.float32) / 255.0
    # Create a dummy 3D volume with a single slice
    volume = img[np.newaxis, ...]  # shape (1, H, W)
    volume_path = os.path.join(project_root, "val_slice.npy")
    np.save(volume_path, volume)
    # Use the latest checkpoint from training
    checkpoint_path = os.path.join(project_root, "model", "model.ckpt")
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    output_path = os.path.join(project_root, "val_slice_pred.npy")
    print(f"Running prediction on slice using checkpoint {checkpoint_path}")
    predict.predict_volume(
        model_path=checkpoint_path,
        volume_path=volume_path,
        output_path=output_path,
        n_classes=3,
        input_size=768,
        use_monai=False,
    )
    pred = np.load(output_path)
    print(f"Prediction shape: {pred.shape}")
    # Save a visual representation (optional)
    # Here we just print unique class values
    print(f"Unique classes in prediction: {np.unique(pred)}")

if __name__ == "__main__":
    main()
