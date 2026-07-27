import pandas as pd
import sys

def compare_addback(csv_path):
    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Check if columns exist
    if 'added_back_count' not in df.columns or 'spatial_added_back_count' not in df.columns:
        print(f"Error: Missing columns in {csv_path}")
        print(f"Columns found: {df.columns.tolist()}")
        return

    # Check for differences
    diff_count = (df['added_back_count'] != df['spatial_added_back_count']).sum()
    
    print(f"--- Comparison Report ---")
    print(f"Total rows: {len(df)}")
    if diff_count == 0:
        print("SUCCESS: 'added_back_count' and 'spatial_added_back_count' are IDENTICAL for all rows.")
    else:
        print(f"WARNING: Found {diff_count} rows with differences!")
        print("\nDiffering rows:")
        print(df[df['added_back_count'] != df['spatial_added_back_count']][['volume', 'added_back_count', 'spatial_added_back_count']])
    
    # Check if they are non-zero
    non_zero = (df['added_back_count'] > 0).sum()
    print(f"\nRows with non-zero add-backs: {non_zero}")

if __name__ == "__main__":
    csv_file = "data/analysis/otsu_pore_recovery_stats.csv"
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    compare_addback(csv_file)
