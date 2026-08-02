import argparse
import sys
import os
import torch

# Ensure project-root package imports work when launched as a script.
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.append(project_root)

from analysis import trainer

def main():
    # Optimization for Tensor Cores (A100)
    torch.set_float32_matmul_precision('medium')

    parser = argparse.ArgumentParser(description="Train U-Net Model")
    
    # Training Loop Parameters
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs to train")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=0.0001, help="Initial learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-2, help="Weight decay")
    
    # Model Parameters
    parser.add_argument("--channels", type=int, default=1, help="Number of input channels")
    parser.add_argument("--classes", type=int, default=3, help="Number of output classes")
    parser.add_argument("--architecture", type=str, default="Unet", help="Model architecture (Unet, UnetPlusPlus, DeepLabV3, etc.)")
    parser.add_argument("--encoder", type=str, default="mit_b0", help="Encoder backbone (mit_b0, resnet34, etc.)")
    parser.add_argument("--encoder-weights", type=str, default="imagenet", help="Pretrained weights (imagenet, None)")
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["plateau", "cosine", "onecycle"], help="Learning rate scheduler")
    parser.add_argument("--loss", type=str, default="tversky", choices=["tversky", "adaptive", "mcc_ce"], help="Loss function")
    parser.add_argument("--tversky-alpha", type=float, nargs="+", default=[0.5, 0.5, 0.3], help="Tversky Loss alpha (for FP). Can be a single float or a list of floats for each class. beta is auto-calculated as 1-alpha.")
    # Data Parameters
    parser.add_argument("--train-dir", type=str, default="data/train", help="Path to training data")
    parser.add_argument("--val-dir", type=str, default="data/val", help="Path to validation data")
    parser.add_argument("--num-workers", type=int, default=8, help="Number of data loading workers")
    
    # Checkpointing
    parser.add_argument("--filename", type=str, default="model", help="Name for the model checkpoint")
    parser.add_argument("--continue-training", action="store_true", help="Continue training from existing checkpoint")

    args = parser.parse_args()

    print(f"Starting training with:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Architecture: {args.architecture} ({args.encoder})")
    print(f"  Data: {args.train_dir} / {args.val_dir}")

    trainer.train_model(
        initial_lr=args.learning_rate,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        epochs=args.epochs,
        n_channels=args.channels,
        n_classes=args.classes,
        architecture=args.architecture,
        encoder_name=args.encoder,
        encoder_weights=args.encoder_weights,
        continue_training=args.continue_training,
        train_folder=args.train_dir,
        val_folder=args.val_dir,
        filename=args.filename,
        num_workers=args.num_workers,
        scheduler=args.scheduler,
        loss_function=args.loss,
        tversky_alpha=args.tversky_alpha
    )

if __name__ == "__main__":
    main()
