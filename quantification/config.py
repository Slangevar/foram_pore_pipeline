"""
Central configuration for the foram pore-quantification package.

Every path resolves from an environment variable when set, otherwise falls back
to a folder beside this file. Point them at your own data before running, either
by exporting the variables or by editing the fallbacks below.

    export FORAM_VOL_DIR=/path/to/clustered_volume     # input label volumes
    export FORAM_NI_INFO=/path/to/"Ni et al info.xlsx"  # voxel-size table
    export FORAM_QUANT_DIR=/path/to/output/quant        # CSV/Excel output
    export FORAM_FIG_DIR=/path/to/output/figures        # figure output

Local-machine example (uncomment/adapt to reproduce the paper without env vars):
    # _REPO = "/Users/you/Porosity"
    # VOL_DIR    = f"{_REPO}/Thinlinc/lu2026-17-19/data/final_foram_state/manual_review/corrected_volume"
    # NI_INFO    = f"{_REPO}/NiSha-paper/Ni et al info.xlsx"
    # ORDER_XLSX = f"{_REPO}/Thinlinc/lu2026-17-19/data/chamber_ordering.xlsx"
"""
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))


def _p(env, *default):
    return os.environ.get(env, os.path.join(HERE, *default))


# ── inputs ───────────────────────────────────────────────────────────────────
# Clustered / manually-corrected label volumes: 3-D uint8 .npy
#   0 = background, 1 = shell, 2+ = per-chamber pore clusters
VOL_DIR = _p("FORAM_VOL_DIR", "data", "clustered_volume")

# Voxel-size table (Ni et al.). Needs columns "CT file name" and "resolution nm".
NI_INFO = _p("FORAM_NI_INFO", "data", "Ni et al info.xlsx")

# Chamber ordering by whorl position (optional; only the chamber-wise figures use it).
ORDER_XLSX = _p("FORAM_ORDER_XLSX", "data", "chamber_ordering.xlsx")

# ── outputs ──────────────────────────────────────────────────────────────────
# CSV / Excel from run_all_quantification.py; also the input dir for the figures.
QUANT_DIR = _p("FORAM_QUANT_DIR", "output", "quantification")

# Figure output.
OUT = _p("FORAM_FIG_DIR", "output", "figures")
OUT_INDIV = os.path.join(OUT, "indiv_figures")   # one title-less figure per subplot

# ── voxel size ───────────────────────────────────────────────────────────────
DEFAULT_VOX_UM = 0.5   # microns/voxel fallback if a sample is not in NI_INFO


def vox_um_for(sample):
    """Per-sample voxel size (µm) from NI_INFO 'resolution nm' (e.g. 500 or 385 nm)."""
    try:
        info = pd.read_excel(NI_INFO)

        def nid(s):
            s = str(s).replace("MOM_", "")
            p = s.split("_")
            return f"{p[0]}_{int(p[1]):02d}" if len(p) == 2 and p[1].isdigit() else s

        info["mid"] = info["CT file name"].astype(str).map(nid)
        row = info[info.mid == nid(sample)]
        if len(row):
            return float(row["resolution nm"].iloc[0]) / 1000.0
        print(f"[vox] {sample} not in {os.path.basename(NI_INFO)}; using {DEFAULT_VOX_UM} µm")
    except Exception as e:
        print(f"[vox] lookup failed ({e}); using {DEFAULT_VOX_UM} µm")
    return DEFAULT_VOX_UM


def ensure_output_dirs():
    for d in (QUANT_DIR, OUT, OUT_INDIV):
        os.makedirs(d, exist_ok=True)
