import argparse
import glob
import os
import sys

# Ensure project-root package imports work when launched as a script.
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.append(project_root)

from analysis import predict

def main():
    parser = argparse.ArgumentParser(description="Predict using U-Net Model")
    
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint (.ckpt)")
    parser.add_argument("--input", type=str, required=True, help="Input file (.npy) or directory")
    parser.add_argument("--output", type=str, default=None, help="Output file or directory (optional)")
    parser.add_argument("--classes", type=int, default=3, help="Number of classes")
    parser.add_argument("--input-size", type=int, default=768, help="Input size for sliding window inference")
    parser.add_argument("--overwrite", action="store_true", help="Force overwrite if prediction already exists")
    
    args = parser.parse_args()
    
    # Handle single file or directory
    if os.path.isdir(args.input):
        input_files = sorted(glob.glob(os.path.join(args.input, "*.npy")))
        if not args.output:
            args.output = args.input.replace("image_volumes", "predicted_volumes")
            if args.output == args.input:
                args.output = args.input + "_pred"
        
        os.makedirs(args.output, exist_ok=True)
        
        print(f"Processing {len(input_files)} files from {args.input}...")
        
        for f in input_files:
            basename = os.path.basename(f)
            out_name = basename.replace(".npy", "_pred.npy")
            out_path = os.path.join(args.output, out_name)
            
            if os.path.exists(out_path) and not args.overwrite:
                print(f" Skipping {basename}: Prediction already exists at {out_path}. Use --overwrite to force.")
                continue
            
            print(f" predicting {basename} -> {out_name}")
            predict.predict_volume(
                model_path=args.model,
                volume_path=f,
                output_path=out_path,
                n_classes=args.classes,
                input_size=args.input_size
            )
            
    else:
        # Single file
        if not args.output:
            args.output = args.input.replace(".npy", "_pred.npy")
            
        if os.path.exists(args.output) and not args.overwrite:
            print(f"Skipping {args.input}: Prediction already exists at {args.output}. Use --overwrite to force.")
            return
            
        print(f"Predicting single file: {args.input}")
        predict.predict_volume(
            model_path=args.model,
            volume_path=args.input,
            output_path=args.output,
            n_classes=args.classes,
            input_size=args.input_size
        )

if __name__ == "__main__":
    main()
