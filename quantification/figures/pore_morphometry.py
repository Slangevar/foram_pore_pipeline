"""
pore_morphometry.py
===================

Per-pore Zingg (1935) shape morphometry, computed directly from the label
volumes. For every connected pore component (voxels >= 2, walls separate the
chambers) it fits the covariance tensor and records the axis-length ratios:

    Elongation = sqrt(lam1/lam2)    # >1.5 -> prolate (rod-like)
    Flatness   = sqrt(lam2/lam3)    # >1.5 -> oblate  (disk-like)

The sqrt turns the PCA variance ratios into axis-LENGTH ratios (variance scales
as length^2), so the 1.5 class boundary matches Zingg's classic 3:2.

Output: <FORAM_QUANT_DIR>/pore_morphometry.csv
Run:    python figures/pore_morphometry.py
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from config import VOL_DIR


def main():
    config.ensure_output_dirs()
    samples = sorted(os.path.splitext(os.path.basename(p))[0]
                     for p in glob.glob(os.path.join(VOL_DIR, "*.npy")))
    rows = []
    for si, s in enumerate(samples):
        try:
            vol = np.load(os.path.join(VOL_DIR, f"{s}.npy"))
        except Exception as e:
            print(f"  skip {s}: {e}", flush=True)
            continue
        lab, n = ndi.label(vol >= 2)                 # connected pore components
        objs = ndi.find_objects(lab)
        for pid, sl in enumerate(objs, 1):
            if sl is None:
                continue
            m = lab[sl] == pid
            coords = np.argwhere(m).astype(np.float64)
            v = len(coords)
            if v < 3:
                continue
            chamber = int(np.bincount(vol[sl][m]).argmax())     # this pore's chamber label
            c = coords - coords.mean(0)
            cov = (c.T @ c) / v
            ev = np.sort(np.linalg.eigvalsh(cov))[::-1]          # lam1 >= lam2 >= lam3
            l1, l2, l3 = ev
            elong = np.sqrt(l1 / l2) if l2 > 1e-6 else np.nan
            flat = np.sqrt(l2 / l3) if l3 > 1e-6 else np.nan
            rows.append((s, chamber, v, float(elong), float(flat)))
        if si % 15 == 0:
            print(f"  ...{si}/{len(samples)}  ({len(rows):,} pores)", flush=True)

    df = pd.DataFrame(rows, columns=["Sample", "Chamber", "n_vox", "Elongation", "Flatness"])
    out = os.path.join(config.QUANT_DIR, "pore_morphometry.csv")
    df.to_csv(out, index=False)
    nfl = df.Flatness.notna().mean() * 100
    print(f"[done] {len(df):,} pores, {df.Sample.nunique()} specimens -> {out}", flush=True)
    print(f"  Flatness defined for {nfl:.0f}% of pores; "
          f"Elongation for {df.Elongation.notna().mean()*100:.0f}%", flush=True)
    print(f"  medians: Elongation={df.Elongation.median():.2f}  "
          f"Flatness={df.Flatness.median():.2f}", flush=True)


if __name__ == "__main__":
    main()
