"""
pore_and_chamber_figures.py
===========================

Publication figures for pore morphometry and chamber-wise pore quantification.

pore_shape  (per-pore morphometry, pooled over all specimens):
    (a) pore-volume distribution
    (b) elongation distribution (L1/L2)     with median + 1.5 class boundary
    (c) flatness distribution   (L2/L3)      with median + 1.5 class boundary
    (d) Zingg 2-D elongation-flatness density with the four class fractions

chamber_metrics  (per-chamber trends along the last whorl, chamber 1..6
with 95% bootstrap CI):
    (a) pores per chamber   (b) pore volume   (c) pore thickness   (d) elongation

Inputs  (in FORAM_QUANT_DIR): pore_morphometry.csv, chamber_summary.csv
        (and FORAM_ORDER_XLSX for whorl ordering)
Outputs (in FORAM_FIG_DIR):   combined PNG/PDF + one figure per panel in indiv_figures/
Run:    python figures/pore_and_chamber_figures.py
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MaxNLocator

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from config import OUT, OUT_INDIV, QUANT_DIR, ORDER_XLSX, vox_um_for

# Reviewer: no figure text below 2 mm printed height. These panels print 4-up
# across the ~170 mm text width, so the smallest text needs >= ~20 pt.
plt.rcParams.update({"font.size": 22, "axes.titlesize": 24, "axes.labelsize": 22,
                     "xtick.labelsize": 22, "ytick.labelsize": 22, "legend.fontsize": 20})
_ANNOT = 20   # in-panel legends / Zingg quadrant + colorbar labels

C_WALL = "#0072B2"; C_PORE = "#D55E00"; CMAP = "viridis"; vox = vox_um_for("MOM_12_01")

config.ensure_output_dirs()
apn = pd.read_csv(os.path.join(QUANT_DIR, "pore_morphometry.csv"))
voxmap = {s: vox_um_for(s) for s in apn.Sample.unique()}
apn["vol_um3"] = apn.n_vox * apn.Sample.map(voxmap) ** 3
EL, FL = apn.Elongation.values, apn.Flatness.values
ELf, FLf = EL[np.isfinite(EL)], FL[np.isfinite(FL)]
both = np.isfinite(EL) & np.isfinite(FL)
el_hi, fl_hi = np.percentile(ELf, 98), np.percentile(FLf, 98)
# Zingg class fractions (threshold 1.5), shown on panel (d)
f_iso = 100 * ((EL <= 1.5) & (FL <= 1.5)).mean(); f_pro = 100 * ((EL > 1.5) & (FL <= 1.5)).mean()
f_obl = 100 * ((EL <= 1.5) & (FL > 1.5)).mean(); f_tri = 100 * ((EL > 1.5) & (FL > 1.5)).mean()


# ---- pore_shape : (a) pore volume  (b) elongation  (c) flatness  (d) Zingg 2-D ----
def ps_vol(ax):
    ax.hist(np.log10(apn.vol_um3), bins=60, color=C_WALL)
    ax.set_xlabel("pore volume (log10 µm³)"); ax.set_ylabel("count")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))   # short ticks -> wider plot
    ax.yaxis.get_offset_text().set_fontsize(_ANNOT)


def ps_el(ax):
    ax.hist(ELf, bins=np.linspace(1, el_hi, 60), color=C_WALL)
    ax.axvline(np.median(ELf), ls="-", color=C_PORE, lw=1.8, label=f"median {np.median(ELf):.2f}")
    ax.axvline(1.5, ls="--", color="k", lw=1.2, label="boundary 1.5")
    ax.set_xlabel("elongation ($L_1/L_2$)"); ax.set_ylabel("count"); ax.set_xlim(1, el_hi)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))   # short ticks -> wider plot
    ax.yaxis.get_offset_text().set_fontsize(_ANNOT)
    ax.set_ylim(top=ax.get_ylim()[1] * 1.34)     # headroom so the legend clears the bars
    ax.legend(fontsize=_ANNOT, loc="upper right", frameon=True, framealpha=1.0, edgecolor="0.7")


def ps_fl(ax):
    ax.hist(FLf, bins=np.linspace(1, fl_hi, 60), color=C_WALL)
    ax.axvline(np.median(FLf), ls="-", color=C_PORE, lw=1.8, label=f"median {np.median(FLf):.2f}")
    ax.axvline(1.5, ls="--", color="k", lw=1.2, label="boundary 1.5")
    ax.set_xlabel("flatness ($L_2/L_3$)"); ax.set_ylabel("count"); ax.set_xlim(1, fl_hi)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))   # short ticks -> wider plot
    ax.yaxis.get_offset_text().set_fontsize(_ANNOT)
    ax.set_ylim(top=ax.get_ylim()[1] * 1.34)     # headroom so the legend clears the bars
    ax.legend(fontsize=_ANNOT, loc="upper right", frameon=True, framealpha=1.0, edgecolor="0.7")


def ps_zingg(ax):
    hb = ax.hexbin(EL[both], FL[both], gridsize=45, cmap=CMAP, bins="log", mincnt=1, extent=(1, el_hi, 1, fl_hi))
    ax.axvline(1.5, ls="--", color="w", lw=1.2); ax.axhline(1.5, ls="--", color="w", lw=1.2)
    ax.set_xlabel("elongation"); ax.set_ylabel("flatness"); ax.set_xlim(1, el_hi); ax.set_ylim(1, fl_hi)
    bb = dict(facecolor="white", alpha=0.75, edgecolor="none", pad=1.0)
    tk = dict(fontsize=19, fontweight="bold", bbox=bb)     # single-line -> fits + >=2 mm
    ax.text(el_hi * 0.70, fl_hi * 0.78, f"triaxial {f_tri:.0f}%", ha="center", va="center", **tk)
    ax.text(el_hi * 0.98, 1.40, f"prolate {f_pro:.0f}%", ha="right", va="center", **tk)
    ax.text(1.25, fl_hi * 0.80, f"oblate {f_obl:.0f}%", ha="center", va="center", rotation=90, **tk)
    ax.text(1.03, 1.10, f"isotropic {f_iso:.0f}%", ha="left", va="center", **tk)
    cb = ax.figure.colorbar(hb, ax=ax); cb.ax.tick_params(labelsize=_ANNOT)


psp = [("(a)", ps_vol), ("(b)", ps_el), ("(c)", ps_fl), ("(d)", ps_zingg)]
fig, axx = plt.subplots(1, 4, figsize=(22, 4.8))
for ax, (lab, fn) in zip(axx, psp):
    fn(ax); ax.set_title(lab, loc="left")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "pore_shape.png"), dpi=220, bbox_inches="tight")
fig.savefig(os.path.join(OUT, "pore_shape.pdf"), bbox_inches="tight"); plt.close(fig)
for i, (lab, fn) in enumerate(psp, 1):
    f, ax = plt.subplots(figsize=(5.4, 4.4)); fn(ax); f.tight_layout()
    f.savefig(os.path.join(OUT_INDIV, f"pore_shape_{i}.png"), dpi=220, bbox_inches="tight")
    f.savefig(os.path.join(OUT_INDIV, f"pore_shape_{i}.pdf"), bbox_inches="tight"); plt.close(f)

# ---- chamber_metrics : count, mean vol, thickness (from chamber_summary) + elongation ----
cs = pd.read_csv(os.path.join(QUANT_DIR, "chamber_summary.csv"))
perch = apn.groupby(["Sample", "Chamber"])[["Elongation", "Flatness"]].mean().reset_index()
cs = cs.merge(perch, on=["Sample", "Chamber"], how="left")
o = pd.read_excel(ORDER_XLSX); COLS = ["1", "2", "3", "4", "5", "6"]; full = o.dropna(subset=COLS)
rows = [(r.Sample, g, int(r[c]) + 1) for _, r in full.iterrows() for g, c in enumerate(COLS, 1)]
L = pd.DataFrame(rows, columns=["Sample", "gpos", "Chamber"]).merge(cs, on=["Sample", "Chamber"], how="left")

panels = [("Num_Pores", "pores per chamber", 1, "(a)", "%.0f"),
          ("Mean_Pore_Volume", "pore volume (µm³)", vox**3, "(b)", "%.0f"),
          ("Mean_LT", "pore thickness (µm)", vox, "(c)", "%.2f"),
          ("Elongation", "elongation", 1, "(d)", "%.2f")]


def draw(ax, col, ylab, sc, fmt=None, label=None):
    rng = np.random.default_rng(0)                # reproducible bootstrap
    m, lo, hi = [], [], []
    for g in range(1, 7):
        v = L.loc[L.gpos == g, col].dropna().values.astype(float) * sc
        boot = np.median(rng.choice(v, size=(2000, len(v)), replace=True), axis=1)
        m.append(np.median(v)); lo.append(np.percentile(boot, 2.5)); hi.append(np.percentile(boot, 97.5))
    ax.fill_between(range(1, 7), lo, hi, color=C_WALL, alpha=0.28)
    ax.plot(range(1, 7), m, "-o", color=C_WALL)
    ax.set_xticks(range(1, 7)); ax.set_xticklabels(["1", "2", "3", "4", "5", "6"])
    ax.set_xlabel("chamber"); ax.set_ylabel(ylab)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    if fmt:
        ax.yaxis.set_major_formatter(FormatStrFormatter(fmt))
    if label:
        ax.set_title(label, loc="left")


fig, axx = plt.subplots(1, 4, figsize=(22, 4.8))
for ax, (col, ylab, sc, lab, fmt) in zip(axx, panels):
    draw(ax, col, ylab, sc, fmt, lab)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "chamber_metrics.png"), dpi=220, bbox_inches="tight")
fig.savefig(os.path.join(OUT, "chamber_metrics.pdf"), bbox_inches="tight"); plt.close(fig)
for i, (col, ylab, sc, lab, fmt) in enumerate(panels, 1):
    f, ax = plt.subplots(figsize=(5.0, 4.4)); draw(ax, col, ylab, sc, fmt); f.tight_layout()
    f.savefig(os.path.join(OUT_INDIV, f"chamber_metrics_{i}.png"), dpi=220, bbox_inches="tight")
    f.savefig(os.path.join(OUT_INDIV, f"chamber_metrics_{i}.pdf"), bbox_inches="tight"); plt.close(f)

# Zingg class fractions
iso = ((EL <= 1.5) & (FL <= 1.5)).mean(); pro = ((EL > 1.5) & (FL <= 1.5)).mean()
obl = ((EL <= 1.5) & (FL > 1.5)).mean(); tri = ((EL > 1.5) & (FL > 1.5)).mean()
print(f"Zingg classes: isotropic {iso*100:.0f}%  prolate {pro*100:.0f}%  oblate {obl*100:.0f}%  triaxial {tri*100:.0f}%")
print("[done] pore_shape + chamber_metrics (+ individuals)")
