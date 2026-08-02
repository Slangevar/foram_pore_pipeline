"""
Run full quantification on all clustered foram samples.

Input:  clustered_volume/*.npy (3D uint8: 0=bg, 1=shell, 2+=chambers)
Output: One Excel file with per-chamber and per-pore metrics,
        LT distributions saved as .npy for reuse.

Usage:
    python run_all_quantification.py
"""

import os
import sys
import time
import numpy as np
import scipy.ndimage as ndi
from scipy.ndimage import distance_transform_edt, binary_erosion
from datetime import datetime

# ── Paths (configure in config.py or via environment variables) ───────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import VOL_DIR as DATA_DIR, QUANT_DIR as OUT_DIR

# ── Metric functions ─────────────────────────────────────────────────────────

def local_thickness_dt(binary_mask):
    dt = distance_transform_edt(binary_mask)
    lt = np.zeros_like(dt)
    lt[binary_mask] = 2 * dt[binary_mask]
    return lt


def calculate_surface_area(binary_mask):
    struct = np.array(
        [[[0,0,0],[0,1,0],[0,0,0]],
         [[0,1,0],[1,1,1],[0,1,0]],
         [[0,0,0],[0,1,0],[0,0,0]]], dtype=bool)
    eroded = binary_erosion(binary_mask, structure=struct)
    surface = binary_mask & ~eroded
    return int(np.sum(surface))


def calculate_sphericity(volume, surface_area):
    if surface_area == 0:
        return 0
    return min(((36 * np.pi * volume**2)**(1/3)) / surface_area, 1.0)


def calculate_cylindricity(pore_mask):
    coords = np.array(np.where(pore_mask)).T
    if len(coords) < 3:
        return 0
    mean = np.mean(coords, axis=0)
    centered = coords - mean
    cov = np.cov(centered.T)
    eigenvals, eigenvecs = np.linalg.eigh(cov)
    idx = eigenvals.argsort()[::-1]
    eigenvecs = eigenvecs[:, idx]
    main_axis = eigenvecs[:, 0]
    proj_matrix = np.eye(3) - np.outer(main_axis, main_axis)
    projected = np.dot(centered, proj_matrix)
    radial_distances = np.linalg.norm(projected, axis=1)
    max_radius = np.max(radial_distances)
    projections = np.dot(centered, main_axis)
    height = np.max(projections) - np.min(projections)
    if height == 0 or max_radius == 0:
        return 0
    ideal_volume = np.pi * max_radius**2 * height
    actual_volume = len(coords)
    if ideal_volume == 0:
        return 0
    return min(actual_volume / ideal_volume, ideal_volume / actual_volume)


def calculate_shape_zingg(pore_mask):
    """Zingg (1935) shape descriptors from the covariance eigenvalues.

    Elongation and Flatness are AXIS-LENGTH ratios, i.e. the sqrt of the
    eigenvalue (variance) ratios -- variance scales as length^2, so taking the
    root puts the 1.5 class boundary on Zingg's 3:2 axis-length scale.
    Returns (elongation, flatness).
      Elongation = sqrt(lam1/lam2): >1.5 = prolate (rod-like)
      Flatness   = sqrt(lam2/lam3): >1.5 = oblate  (disk-like)
    """
    coords = np.array(np.where(pore_mask)).T.astype(float)
    if len(coords) < 3:
        return 1.0, 1.0
    mean = np.mean(coords, axis=0)
    centered = coords - mean
    cov = np.cov(centered.T)
    eigenvals = np.sort(np.linalg.eigvalsh(cov))[::-1]
    lam1, lam2, lam3 = eigenvals
    if lam2 <= 0 or lam3 <= 0:
        return 1.0, 1.0
    return float(np.sqrt(lam1 / lam2)), float(np.sqrt(lam2 / lam3))


# ── Process one sample ───────────────────────────────────────────────────────

def process_sample(npy_path, lt_dir):
    """Process one sample, return (chamber_rows, pore_rows)."""
    name = os.path.basename(npy_path).replace('.npy', '')
    data = np.load(npy_path)

    if data.ndim != 3:
        print(f"  [SKIP] {name}: expected 3D, got {data.shape}")
        return [], []

    class_labels = np.unique(data)
    class_labels = class_labels[class_labels >= 2]  # skip bg(0) and shell(1)

    chamber_rows = []
    pore_rows = []

    for chamber_label in class_labels:
        ch = int(chamber_label)
        mask = (data == chamber_label).astype(bool)
        total_volume = int(np.sum(mask))

        if total_volume == 0:
            continue

        t0 = time.time()

        # Local thickness — load from cache or compute in memory
        lt_path = os.path.join(lt_dir, f'{name}_chamber_{ch}_LT.npy')
        if os.path.exists(lt_path):
            try:
                lt = np.load(lt_path)
                if lt.size == 0 or lt.max() == 0:
                    raise ValueError("corrupt cache")
            except Exception:
                lt = local_thickness_dt(mask)
        else:
            lt = local_thickness_dt(mask)

        lt_nonzero = lt[lt > 0]

        # Connected components + bounding boxes for fast per-pore crops
        labeled, num_pores = ndi.label(mask)
        slices = ndi.find_objects(labeled)

        # Per-pore analysis (skip dust < 5 voxels)
        pore_vols = []
        pore_sas = []
        pore_sphs = []
        pore_cyls = []
        pore_mean_lts = []
        pore_elongs = []
        pore_flats = []

        for pore_id in range(1, num_pores + 1):
            sl = slices[pore_id - 1]
            if sl is None:
                continue

            # Crop to bounding box (+1 padding for erosion)
            pad_sl = tuple(
                slice(max(0, s.start - 1), min(mask.shape[i], s.stop + 1))
                for i, s in enumerate(sl)
            )
            crop_labeled = labeled[pad_sl]
            crop_lt = lt[pad_sl]
            pore_mask = (crop_labeled == pore_id)

            vol = int(np.sum(pore_mask))
            if vol < 5:
                continue

            pore_lt_vals = crop_lt[pore_mask]
            pore_lt_nz = pore_lt_vals[pore_lt_vals > 0]
            if pore_lt_nz.size == 0:
                continue

            sa = calculate_surface_area(pore_mask)
            sph = calculate_sphericity(vol, sa)
            cyl = calculate_cylindricity(pore_mask)
            elong, flat = calculate_shape_zingg(pore_mask)

            pore_vols.append(vol)
            pore_sas.append(sa)
            pore_sphs.append(sph)
            pore_cyls.append(cyl)
            pore_mean_lts.append(float(np.mean(pore_lt_nz)))
            pore_elongs.append(elong)
            pore_flats.append(flat)

            pore_rows.append({
                'Sample': name,
                'Chamber': ch,
                'Pore_ID': pore_id,
                'Volume': vol,
                'Surface_Area': sa,
                'Relative_Sphericity': round(sph, 4),
                'Cylindricity': round(cyl, 4),
                'Elongation': round(elong, 4),
                'Flatness': round(flat, 4),
                'Max_LT': round(float(np.max(pore_lt_nz)), 2),
                'Min_LT': round(float(np.min(pore_lt_nz)), 2),
                'Mean_LT': round(float(np.mean(pore_lt_nz)), 2),
                'Std_LT': round(float(np.std(pore_lt_nz)), 2),
            })

        elapsed = time.time() - t0
        n_valid = len(pore_vols)

        # Chamber summary
        chamber_rows.append({
            'Sample': name,
            'Chamber': ch,
            'Num_Pores': n_valid,
            'Total_Volume': total_volume,
            'Mean_Pore_Volume': round(np.mean(pore_vols), 1) if pore_vols else 0,
            'Std_Pore_Volume': round(np.std(pore_vols), 1) if pore_vols else 0,
            'Max_Pore_Volume': max(pore_vols) if pore_vols else 0,
            'Mean_Surface_Area': round(np.mean(pore_sas), 1) if pore_sas else 0,
            'Mean_Relative_Sphericity': round(np.mean(pore_sphs), 4) if pore_sphs else 0,
            'Std_Relative_Sphericity': round(np.std(pore_sphs), 4) if pore_sphs else 0,
            'Mean_Cylindricity': round(np.mean(pore_cyls), 4) if pore_cyls else 0,
            'Std_Cylindricity': round(np.std(pore_cyls), 4) if pore_cyls else 0,
            'Mean_Elongation': round(np.mean(pore_elongs), 4) if pore_elongs else 1,
            'Std_Elongation': round(np.std(pore_elongs), 4) if pore_elongs else 0,
            'Mean_Flatness': round(np.mean(pore_flats), 4) if pore_flats else 1,
            'Std_Flatness': round(np.std(pore_flats), 4) if pore_flats else 0,
            'Max_LT': round(float(np.max(lt_nonzero)), 2) if lt_nonzero.size > 0 else 0,
            'Min_LT': round(float(np.min(lt_nonzero)), 2) if lt_nonzero.size > 0 else 0,
            'Mean_LT': round(float(np.mean(lt_nonzero)), 2) if lt_nonzero.size > 0 else 0,
            'Std_LT': round(float(np.std(lt_nonzero)), 2) if lt_nonzero.size > 0 else 0,
            'Processing_Time_s': round(elapsed, 1),
        })

        # Free memory before next chamber
        del lt, lt_nonzero, labeled, slices, mask
        import gc; gc.collect()

        print(f"    Ch {ch}: {n_valid} pores, vol={total_volume:,}, "
              f"sph={chamber_rows[-1]['Mean_Relative_Sphericity']:.3f}, "
              f"cyl={chamber_rows[-1]['Mean_Cylindricity']:.3f}, "
              f"LT={chamber_rows[-1]['Mean_LT']:.2f}  ({elapsed:.1f}s)")

    return chamber_rows, pore_rows


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    import csv

    os.makedirs(OUT_DIR, exist_ok=True)
    lt_dir = os.path.join(OUT_DIR, 'local_thickness')
    os.makedirs(lt_dir, exist_ok=True)

    npy_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.npy')])
    print(f"Found {len(npy_files)} samples in {DATA_DIR}")
    print(f"Output: {OUT_DIR}")
    print(f"LT cache: {lt_dir}")
    print()

    all_chamber_rows = []
    all_pore_rows = []
    total_t0 = time.time()

    # Check for existing progress (resume support)
    chamber_csv = os.path.join(OUT_DIR, 'chamber_summary.csv')
    pore_csv = os.path.join(OUT_DIR, 'all_pores.csv')
    done_samples = set()

    if os.path.exists(chamber_csv):
        with open(chamber_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                all_chamber_rows.append(row)
                done_samples.add(row['Sample'])
        print(f"Resuming: {len(done_samples)} samples already processed")

    if os.path.exists(pore_csv):
        with open(pore_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                all_pore_rows.append(row)

    for i, npy_file in enumerate(npy_files):
        name = npy_file.replace('.npy', '')
        if name in done_samples:
            print(f"[{i+1}/{len(npy_files)}] {name} — already done, skipping")
            continue

        print(f"[{i+1}/{len(npy_files)}] {name}")
        npy_path = os.path.join(DATA_DIR, npy_file)

        try:
            ch_rows, p_rows = process_sample(npy_path, lt_dir)
            all_chamber_rows.extend(ch_rows)
            all_pore_rows.extend(p_rows)

            # Save progress after each sample (resume-safe)
            if ch_rows:
                write_header = not os.path.exists(chamber_csv)
                with open(chamber_csv, 'a', newline='') as f:
                    w = csv.DictWriter(f, fieldnames=ch_rows[0].keys())
                    if write_header:
                        w.writeheader()
                    w.writerows(ch_rows)

            if p_rows:
                write_header = not os.path.exists(pore_csv)
                with open(pore_csv, 'a', newline='') as f:
                    w = csv.DictWriter(f, fieldnames=p_rows[0].keys())
                    if write_header:
                        w.writeheader()
                    w.writerows(p_rows)

        except Exception as e:
            print(f"  [ERROR] {name}: {e}")
            continue

    total_elapsed = time.time() - total_t0

    # Write final Excel
    try:
        import pandas as pd
        excel_path = os.path.join(OUT_DIR, 'quantification_all_samples.xlsx')
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            pd.DataFrame(all_chamber_rows).to_excel(
                writer, sheet_name='Chamber_Summary', index=False)
            pd.DataFrame(all_pore_rows).to_excel(
                writer, sheet_name='All_Pores', index=False)
        print(f"\nExcel saved: {excel_path}")
    except Exception as e:
        print(f"\nExcel write failed ({e}), but CSVs are saved.")

    print(f"\nDone! {len(npy_files)} samples, {len(all_chamber_rows)} chambers, "
          f"{len(all_pore_rows)} pores")
    print(f"Total time: {total_elapsed/60:.1f} min")
    print(f"Files:")
    print(f"  {chamber_csv}")
    print(f"  {pore_csv}")
    print(f"  LT arrays: {lt_dir}/")


if __name__ == '__main__':
    main()
