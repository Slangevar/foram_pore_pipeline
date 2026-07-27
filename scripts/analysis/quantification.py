#!/usr/bin/env python3
"""
Per-chamber pore quantification adapted from Behnaz's quantification.ipynb.

Input: clustered_volume/*.npy files (3-D uint8, voxel value = class label)
    0  = background
    1  = shell  (skipped — not a pore cluster)
    2+ = chamber pore clusters

For each volume and each chamber (class >= 2) the script computes:
    - Total pore voxel count
    - Number of connected pore components
    - Max / Min / Std / Mean pore-component volume
    - Max / Min / Std / Mean local thickness (porespy)

Outputs (written to --out-dir):
    quantification_results.csv   — per-file-per-chamber metrics
    <name>_class_<id>_LT.npy    — local thickness distribution arrays
    <name>_class_<id>_LT.png    — KDE plot per chamber

Usage:
    python scripts/analysis/quantification.py \
        --data-dir  data/final_foram_state/clustered_volume \
        --out-dir   data/analysis/quantification \
        --volumes   MOM_12_01          # comma-separated stems; omit for all
"""

import argparse
import os
import time
import glob
from datetime import datetime

import numpy as np
import scipy.ndimage as ndi
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    import porespy as ps
except ImportError:
    raise ImportError("porespy is required: pip install porespy")

try:
    import seaborn as sns
except ImportError:
    raise ImportError("seaborn is required: pip install seaborn")

try:
    import pandas as pd
except ImportError:
    raise ImportError("pandas is required: pip install pandas")


def compute_metrics(binary_data):
    """
    Compute local thickness and pore-component statistics for a binary mask.
    Mirrors Behnaz's compute_metrics() from quantification.ipynb.
    """
    binary_data = (binary_data > 0)
    if not binary_data.any():
        return {
            "Total_Volume": 0, "Num_Pores": 0,
            "Max_Pore_Volume": 0, "Min_Pore_Volume": 0,
            "Std_Pore_Volume": 0.0, "Mean_Pore_Volume": 0.0,
            "Max_LT": 0, "Min_LT": 0, "Std_LT": 0.0, "Mean_LT": 0.0,
            "LT_Distribution": np.array([]),
        }

    local_thickness = ps.filters.local_thickness(binary_data, method='dt')
    lt_nonzero = local_thickness[local_thickness > 0]

    total_volume = int(np.sum(binary_data))
    labeled_pores, num_pores = ndi.label(binary_data)
    pore_volumes = [int(np.sum(labeled_pores == i)) for i in range(1, num_pores + 1)]

    return {
        "Total_Volume":    total_volume,
        "Num_Pores":       num_pores,
        "Max_Pore_Volume": max(pore_volumes) if pore_volumes else 0,
        "Min_Pore_Volume": min(pore_volumes) if pore_volumes else 0,
        "Std_Pore_Volume": float(np.std(pore_volumes)) if pore_volumes else 0.0,
        "Mean_Pore_Volume": float(np.mean(pore_volumes)) if pore_volumes else 0.0,
        "Max_LT":  float(np.max(lt_nonzero))  if lt_nonzero.size > 0 else 0,
        "Min_LT":  float(np.min(lt_nonzero))  if lt_nonzero.size > 0 else 0,
        "Std_LT":  float(np.std(lt_nonzero))  if lt_nonzero.size > 0 else 0.0,
        "Mean_LT": float(np.mean(lt_nonzero)) if lt_nonzero.size > 0 else 0.0,
        "LT_Distribution": lt_nonzero,
    }


def process_volume(npy_path, out_dir):
    file_name = os.path.basename(npy_path)
    name = file_name.replace('.npy', '')
    print(f'\n[{name}]', flush=True)

    data = np.load(npy_path)
    if data.ndim != 3:
        print(f'  [SKIP] Expected 3-D array, got shape {data.shape}')
        return []

    # Classes >= 2 are chamber pore clusters; 0=background, 1=shell (skip both)
    class_labels = np.unique(data)
    class_labels = class_labels[class_labels >= 2]
    print(f'  Chambers found: {class_labels.tolist()}', flush=True)

    rows = []
    for class_label in class_labels:
        int_label = int(class_label)
        print(f'  Chamber {int_label} ...', flush=True)
        t0 = time.time()

        class_data = (data == class_label).astype(np.uint8)
        metrics = compute_metrics(class_data)

        # Save LT distribution array
        lt_values = metrics["LT_Distribution"]
        lt_values = lt_values[lt_values > 0]
        lt_npy_path = os.path.join(out_dir, f'{name}_class_{int_label}_LT.npy')
        np.save(lt_npy_path, lt_values)

        # Save KDE plot
        if lt_values.size > 0:
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.kdeplot(lt_values, fill=True, bw_adjust=0.5, ax=ax)
            ax.set_title(f'Local Thickness Distribution\n{name} — Chamber {int_label}')
            ax.set_xlabel('Local Thickness')
            ax.set_ylabel('Density')
            ax.grid(True)
            plot_path = os.path.join(out_dir, f'{name}_class_{int_label}_LT.png')
            fig.savefig(plot_path, bbox_inches='tight')
            plt.close(fig)

        elapsed = time.time() - t0
        print(f'    total_vox={metrics["Total_Volume"]:,}  '
              f'num_pores={metrics["Num_Pores"]}  '
              f'mean_LT={metrics["Mean_LT"]:.2f}  '
              f'({elapsed:.1f}s)', flush=True)

        rows.append({
            "File_Name":            f'{name}_class_{int_label}',
            "Timestamp":            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "Processing_Time_s":    round(elapsed, 2),
            "Pore_Total_Volume":    metrics["Total_Volume"],
            "Pore_Num":             metrics["Num_Pores"],
            "Pore_Max_Volume":      metrics["Max_Pore_Volume"],
            "Pore_Min_Volume":      metrics["Min_Pore_Volume"],
            "Pore_Std_Volume":      metrics["Std_Pore_Volume"],
            "Pore_Mean_Volume":     metrics["Mean_Pore_Volume"],
            "Pore_Max_LT":          metrics["Max_LT"],
            "Pore_Min_LT":          metrics["Min_LT"],
            "Pore_Std_LT":          metrics["Std_LT"],
            "Pore_Mean_LT":         metrics["Mean_LT"],
        })

    return rows


def main():
    ap = argparse.ArgumentParser(
        description='Per-chamber pore quantification using porespy local thickness'
    )
    ap.add_argument('--data-dir', required=True,
                    help='Folder containing clustered_volume *.npy files')
    ap.add_argument('--out-dir', default='data/analysis/quantification',
                    help='Output directory for CSV, LT arrays, and plots')
    ap.add_argument('--volumes', default='',
                    help='Comma-separated volume stems (no .npy). '
                         'If empty, processes all files in --data-dir.')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.volumes.strip():
        names     = [v.strip() for v in args.volumes.split(',') if v.strip()]
        npy_files = [os.path.join(args.data_dir, f'{n}.npy') for n in names]
    else:
        npy_files = sorted(glob.glob(os.path.join(args.data_dir, '*.npy')))

    print(f'Processing {len(npy_files)} volume(s) from {args.data_dir}')

    all_rows = []
    for f in npy_files:
        if not os.path.isfile(f):
            print(f'[WARN] File not found: {f}')
            continue
        rows = process_volume(f, args.out_dir)
        all_rows.extend(rows)

    csv_path = os.path.join(args.out_dir, 'quantification_results.csv')
    df = pd.DataFrame(all_rows)
    df.to_csv(csv_path, index=False)
    print(f'\nResults saved to: {csv_path}')
    print('Done.')


if __name__ == '__main__':
    main()
