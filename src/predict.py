import os
import glob
import numpy as np
import torch

from . import model, utils

def manual_sliding_window_3d(volume_tensor, model_fn, roi_size=(768, 768), spatial_dim=0, batch_size=16):
    """
    Manual sliding window inference for 3D volumes.
    
    Args:
        volume_tensor: (1, C, D, H, W) tensor
        model_fn: Function that takes (B, C, H, W) and returns (B, n_classes, H, W)
        roi_size: (H, W) size of the sliding window
        spatial_dim: Which dimension to slice along (0=D, 1=H, 2=W)
        batch_size: Number of slices to process at once
    
    Returns:
        (1, n_classes, D, H, W) tensor of probabilities
    """
    # Use model device for inference, keep full volume on CPU
    model_device = next(model_fn.parameters()).device if hasattr(model_fn, 'parameters') else torch.device('cuda')
    device = volume_tensor.device
    _, C, D, H, W = volume_tensor.shape
    
    # Determine output shape - we need to run one slice to get n_classes
    if spatial_dim == 0:  # Axial slices (along D)
        test_slice = volume_tensor[:, :, 0:1, :, :].squeeze(2)  # (1, C, H, W)
        with torch.no_grad():
            test_out = model_fn(test_slice.to(model_device))  # (1, n_classes, H, W)
        n_classes = test_out.shape[1]
        output = torch.zeros((1, n_classes, D, H, W), device=device, dtype=torch.float32)
        
        # Process slices in batches
        for start_idx in range(0, D, batch_size):
            end_idx = min(start_idx + batch_size, D)
            batch_slices = volume_tensor[:, :, start_idx:end_idx, :, :]  # (1, C, batch, H, W)
            batch_slices = batch_slices.squeeze(0).permute(1, 0, 2, 3)  # (batch, C, H, W)
            
            with torch.no_grad():
                batch_out = model_fn(batch_slices.to(model_device))  # (batch, n_classes, H, W)
            
            output[:, :, start_idx:end_idx, :, :] = batch_out.to(device).unsqueeze(0).permute(0, 2, 1, 3, 4)
            
    elif spatial_dim == 1:  # Coronal slices (along H)
        test_slice = volume_tensor[:, :, :, 0:1, :].squeeze(3).permute(0, 1, 3, 2)  # (1, C, W, D)
        # Resize to roi_size for model
        test_slice_resized = torch.nn.functional.interpolate(test_slice, size=roi_size, mode='bilinear')
        with torch.no_grad():
            test_out = model_fn(test_slice_resized.to(model_device))
        n_classes = test_out.shape[1]
        output = torch.zeros((1, n_classes, D, H, W), device=device, dtype=torch.float32)
        
        for start_idx in range(0, H, batch_size):
            end_idx = min(start_idx + batch_size, H)
            batch_slices = volume_tensor[:, :, :, start_idx:end_idx, :]  # (1, C, D, batch, W)
            batch_slices = batch_slices.squeeze(0).permute(2, 0, 3, 1)  # (batch, C, W, D)
            batch_slices = torch.nn.functional.interpolate(batch_slices, size=roi_size, mode='bilinear')
            
            with torch.no_grad():
                batch_out = model_fn(batch_slices.to(model_device))  # (batch, n_classes, roi_H, roi_W)
            
            # Resize back and place
            batch_out = torch.nn.functional.interpolate(batch_out, size=(W, D), mode='bilinear')
            output[:, :, :, start_idx:end_idx, :] = batch_out.to(device).unsqueeze(0).permute(0, 2, 4, 1, 3)
            
    else:  # spatial_dim == 2, Sagittal slices (along W)
        test_slice = volume_tensor[:, :, :, :, 0:1].squeeze(4).permute(0, 1, 3, 2)  # (1, C, H, D)
        test_slice_resized = torch.nn.functional.interpolate(test_slice, size=roi_size, mode='bilinear')
        with torch.no_grad():
            test_out = model_fn(test_slice_resized.to(model_device))
        n_classes = test_out.shape[1]
        output = torch.zeros((1, n_classes, D, H, W), device=device, dtype=torch.float32)
        
        for start_idx in range(0, W, batch_size):
            end_idx = min(start_idx + batch_size, W)
            batch_slices = volume_tensor[:, :, :, :, start_idx:end_idx]  # (1, C, D, H, batch)
            batch_slices = batch_slices.squeeze(0).permute(3, 0, 2, 1)  # (batch, C, H, D)
            batch_slices = torch.nn.functional.interpolate(batch_slices, size=roi_size, mode='bilinear')
            
            with torch.no_grad():
                batch_out = model_fn(batch_slices.to(model_device))  # (batch, n_classes, roi_H, roi_W)
            
            # Resize back and place
            batch_out = torch.nn.functional.interpolate(batch_out, size=(H, D), mode='bilinear')
            output[:, :, :, :, start_idx:end_idx] = batch_out.to(device).unsqueeze(0).permute(0, 2, 4, 3, 1)
    
    return output

def predict_single_slice(image_slice, n_channels=1, n_classes=3, device=None, model_instance=None):
    """
    Helper to predict a single 2D slice (mostly for visualization/testing).
    """
    if device is None:
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
    # Prepare input: (1, C, H, W)
    if len(image_slice.shape) == 2:
        X = image_slice[None, None, :, :] # (1, 1, H, W)
    else:
        # Assuming (H, W, C) or similar, ensure it matches model input
        X = image_slice[None, ...]
        
    X = torch.tensor(X.astype(np.float32)).to(device)
    
    if model_instance is None:
        # Load default if not provided (inefficient for loops, but ok for single calls)
        model_instance = model.SegmentationModel(num_channels=n_channels, num_classes=n_classes).to(device)
        model_instance.eval()
        
    with torch.no_grad():
        y_prob = model_instance(X).cpu().detach().numpy() # (1, n_classes, H, W)

    y_pred = np.argmax(y_prob[0], axis=0).astype(np.uint8) # (H, W)
    
    # Global 'Black Pixel' Masking - treat low-value pixels as background (Class 0, Red)
    # Using a small threshold (e.g., 20/255 ≈ 0.08) for robustness
    bg_threshold = 0.08
    if image_slice.ndim == 3:
        mask_background = (np.mean(image_slice, axis=0) < bg_threshold)
    else:
        mask_background = (image_slice < bg_threshold)
        
    y_pred[mask_background] = 0
    
    return y_pred


def predict_volume(
    model_path: str, volume_path: str, output_path: str,  n_classes: int = 3, input_size: int = 768, use_monai: bool = False
):
    """
    Predicts segmentation for a 3D volume using multi-view consensus.
    
    Args:
        model_path: Path to the trained checkpoint.
        volume_path: Path to the input .npy volume.
        output_path: Path to save the prediction.
        n_classes: Number of classes.
        input_size: ROI size for sliding window (H, W).
        use_monai: Whether to use monai.inferers.SliceInferer (may hang on macOS).
    """
    # Get device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        
    print(f"Prediction using device: {device}")

    # Load model
    print(f"Loading model from {model_path}...")
    try:
        loaded_model = model.SegmentationModel.load_from_checkpoint(checkpoint_path=model_path)
    except Exception as e:
        print(f"Warning: Could not load from checkpoint ({e}), initing new model (check weights!)")
        loaded_model = model.SegmentationModel(num_classes=n_classes)
        
    loaded_model.to(device)
    loaded_model.eval()

    # Load Volume: (D, H, W) -> assumed grayscale
    print(f"Loading volume {volume_path}...")
    volume_np = np.load(volume_path).astype(np.float32) / 255.0
    
    # Ensure massive 3D accumulations happen heavily on CPU RAM out-of-core
    accum_device = torch.device('cpu')

    # Create input tensor: (1, 1, D, H, W)
    volume_tensor = torch.tensor(volume_np[None, None, ...], device=accum_device)

    # Soft Voting Accumulator: (1, n_classes, D, H, W)
    # We sum probabilities directly, then argmax at the very end.
    prob_sum = torch.zeros(
        (1, n_classes, *volume_np.shape), device=accum_device, dtype=torch.float32
    )

    # 3 Orthogonal Views x 4 Rotations = 12 Views
    dims = [0, 1, 2]
    
    # Mapping slice_dim -> rotation_plane (indices in N,C,D,H,W 5D tensor)
    # 0 (Axial D)    -> Plane (H, W) -> dims [3, 4]
    # 1 (Coronal H)  -> Plane (D, W) -> dims [2, 4]
    # 2 (Sagittal W) -> Plane (D, H) -> dims [2, 3]
    rot_planes = {0: [3, 4], 1: [2, 4], 2: [2, 3]}

    if use_monai:
        try:
            from monai.inferers import SliceInferer
            print("Using MONAI SliceInferer...")
        except ImportError:
            print("Error: MONAI not found. Falling back to manual implementation.")
            use_monai = False
            
    total_views = 0
    for d in dims:
        for k in [0, 1, 2, 3]: # 4 Rotations (0, 90, 180, 270)
            print(f"Predicting dim {d} | Rotation {k*90}° ...", flush=True)
            
            # 1. Rotate Volume
            # We use rot90 which is efficient and lossless
            rot_dims = rot_planes[d]
            volume_rotated = torch.rot90(volume_tensor, k, dims=rot_dims)
            
            # 2. Inference
            if use_monai:
                from monai.inferers import SliceInferer
                inferer = SliceInferer(
                    roi_size=(input_size, input_size),
                    sw_batch_size=16, 
                    spatial_dim=d,
                    overlap=0.25,
                    mode="gaussian",
                    padding_mode="reflect"
                )
                with torch.no_grad():
                    view_probs_rot = inferer(volume_rotated, loaded_model)
            else:
                view_probs_rot = manual_sliding_window_3d(
                    volume_rotated, 
                    loaded_model, 
                    roi_size=(input_size, input_size),
                    spatial_dim=d,
                    batch_size=16
                )
            
            # 3. Inverse Rotate Probabilities
            # We must rotate back by -k (or 4-k) to align with original
            view_probs_aligned = torch.rot90(view_probs_rot, -k, dims=rot_dims)
            
            # 4. Accumulate Soft Probabilities
            prob_sum += view_probs_aligned
            total_views += 1

    # Final Argmax
    print("Post-processing (Soft Voting)...", flush=True)
    
    # We don't need to divide by total_views for argmax since it's a constant positive scalar!
    # Move to CPU *first* so the massive argmax computation happens on CPU RAM instead of VRAM
    prob_sum_cpu = prob_sum.cpu()
    del prob_sum # Free up GPU memory immediately
    torch.cuda.empty_cache() # Purge the cache for the next volume in the batch
    
    final_pred = torch.argmax(prob_sum_cpu, dim=1).squeeze(0).numpy().astype(np.uint8) # (D, H, W)
    del prob_sum_cpu
    # Global 'Black Pixel' Masking
    mask_background = (volume_np < 0.08)
    final_pred[mask_background] = 0

    print(f"Saving to {output_path}...")
    np.save(output_path, final_pred)