"""
local_thickness_3d.py
=====================

Interactive 3-D local-thickness visualisations for one foram, from a
manually-corrected label volume. Five scenes:

  1. Shell wall CUT through the middle, cut face coloured by wall local
     thickness -> read the wall thickness directly in cross-section.
  2. Whole shell, semi-transparent, coloured by wall local thickness.
  3. Whole foram: all pores coloured by chamber (editor palette).
  4. Whole foram: all pores coloured by their own local thickness.
  5. Transparent whole shell + one chamber's pores highlighted by local thickness.

Local thickness = diameter of the largest inscribed sphere (Hildebrand 1997),
computed with SciPy distance transforms. Surfaces are Gaussian-smoothed +
Gouraud-shaded.

Output: <FORAM_FIG_DIR>/thickness_3d_report.html  (self-contained; drag to
rotate, scroll to zoom).
Run:    python figures/local_thickness_3d.py --sample MOM_12_01 --chamber 14
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from scipy import ndimage as ndi
from skimage import measure
import plotly.graph_objects as go

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from config import VOL_DIR, ORDER_XLSX, vox_um_for

VOX_UM = config.DEFAULT_VOX_UM  # microns per voxel; set PER SAMPLE in main() from Ni info


def nth_from_last_label(sample, n):
    """Data label of the n-th chamber counting back from the last (outermost).

    chamber_ordering.xlsx: column "1" = last/outermost chamber, ... "6" = 6th
    from the end. Each value is the pore-count rank, so data label = value + 1.
    """
    try:
        import pandas as pd
        o = pd.read_excel(ORDER_XLSX)
        row = o[o.Sample == sample]
        val = row[str(n)].iloc[0]
        if pd.notna(val):
            return int(val) + 1
    except Exception as e:
        print("[order] lookup failed:", e)
    return None


# chamber palette (matches the manual-correction editor)
PALETTE = ['#E6194B', '#3CB44B', '#4363D8', '#F58231', '#911EB4', '#42D4F4',
           '#F032E6', '#BFEF45', '#FABED4', '#469990', '#DCBEFF', '#9A6324',
           '#800000', '#AAFFC3', '#808000', '#FFD8B1', '#000075', '#A9A9A9',
           '#E6BEFF', '#FF6F61', '#1B998B', '#C0C0C0']


def local_thickness(mask, rmax=None):
    """Hildebrand local thickness (in voxels)."""
    edt = ndi.distance_transform_edt(mask).astype(np.float32)
    if rmax is None:
        rmax = int(np.ceil(edt.max()))
    lt = np.zeros_like(edt)
    for r in range(1, rmax + 1):
        Sr = edt >= r
        if not Sr.any():
            break
        d = ndi.distance_transform_edt(~Sr)
        lt[d <= r] = 2.0 * r
    leftover = mask & (lt == 0)
    lt[leftover] = np.maximum(2.0 * edt[leftover], 2.0)
    return lt


def smooth_field(mask, sigma):
    return ndi.gaussian_filter(mask.astype(np.float32), sigma)


def mc(field, level=0.5, step=1):
    if field.max() < level:
        return None
    v, f, _, _ = measure.marching_cubes(field, level=level, step_size=step)
    return v, f


def sample_at(vol, verts):
    idx = np.clip(np.round(verts).astype(int), 0, np.array(vol.shape) - 1)
    return vol[idx[:, 0], idx[:, 1], idx[:, 2]]


def mesh_trace(verts, faces, scale, intensity=None, colorscale="Turbo",
               cmin=None, cmax=None, color=None, opacity=1.0, name="",
               colorbar_title=None, showscale=True, showlegend=False):
    kw = dict(
        x=verts[:, 2] * scale, y=verts[:, 1] * scale, z=verts[:, 0] * scale,
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        opacity=opacity, name=name, flatshading=False, showlegend=showlegend,
        lighting=dict(ambient=0.5, diffuse=0.85, specular=0.2, roughness=0.85, fresnel=0.1),
        lightposition=dict(x=1500, y=1500, z=2500),
    )
    if intensity is not None:
        kw.update(intensity=intensity, intensitymode="vertex", colorscale=colorscale,
                  cmin=cmin, cmax=cmax, showscale=showscale,
                  colorbar=dict(title=colorbar_title or "µm", len=0.6, thickness=16))
    else:
        kw.update(color=color, showscale=False)
    return go.Mesh3d(**kw)


def layout(title, eye=(1.5, 1.4, 1.1)):
    ax = dict(showbackground=False, showticklabels=False, title="", showgrid=False, zeroline=False)
    return dict(title=dict(text=title, x=0.5, font=dict(size=17, color="#e8eaf0")),
                scene=dict(xaxis=ax, yaxis=ax, zaxis=ax, aspectmode="data",
                           camera=dict(eye=dict(x=eye[0], y=eye[1], z=eye[2]))),
                paper_bgcolor="#11151d", font=dict(color="#cdd3df"),
                margin=dict(l=0, r=0, t=44, b=0), height=640)


# ---------------------------------------------------------------------------
def shell_lt_ds(vol, ds=2):
    shd = ndi.zoom(smooth_field(vol == 1, 1.0), 1.0 / ds, order=1) > 0.4
    lt = local_thickness(shd)
    return shd, lt


def scene_shell_cut(vol, ds=2, axis=2):
    shd, lt = shell_lt_ds(vol, ds)
    f = smooth_field(shd, 0.8)
    cc = shd.shape[axis] // 2
    sl = [slice(None)] * 3
    sl[axis] = slice(cc, None)
    fclip = f.copy()
    fclip[tuple(sl)] = 0.0                      # remove one half -> cut face appears
    m = mc(fclip, 0.5, step=2)
    v, fc = m
    lt_v = sample_at(lt, v) * ds * VOX_UM
    cmax = float(np.percentile(lt_v[lt_v > 0], 97))
    tr = mesh_trace(v, fc, ds * VOX_UM, intensity=lt_v, colorscale="Turbo",
                    cmin=0, cmax=cmax, name="shell (cut)", colorbar_title="wall<br>thickness (µm)")
    # look straight at the cut face (cut-plane normal = the removed axis)
    eyes = {0: (0.25, 0.25, 2.0), 1: (0.25, 2.0, 0.25), 2: (2.0, 0.25, 0.25)}
    fig = go.Figure([tr])
    fig.update_layout(**layout("1 · Shell cut through the middle — wall thickness in cross-section",
                               eye=eyes.get(axis, (1.5, 1.4, 1.1))))
    return fig


def scene_shell_transparent(vol, ds=2):
    shd, lt = shell_lt_ds(vol, ds)
    m = mc(smooth_field(shd, 0.9), 0.5, step=3)
    v, fc = m
    lt_v = sample_at(lt, v) * ds * VOX_UM
    cmax = float(np.percentile(lt_v[lt_v > 0], 97))
    tr = mesh_trace(v, fc, ds * VOX_UM, intensity=lt_v, colorscale="Turbo",
                    cmin=0, cmax=cmax, opacity=0.55, name="shell", colorbar_title="wall<br>thickness (µm)")
    fig = go.Figure([tr]); fig.update_layout(**layout("2 · Whole shell (semi-transparent) — wall thickness"))
    return fig


def scene_pores_whole(vol, ds_shell=3):
    """Whole foram: pores coloured by CHAMBER (one colour each, editor palette)."""
    print("[pores] colouring pores by chamber ...")
    labels = sorted(int(l) for l in np.unique(vol) if l >= 2)
    data = []
    shd = ndi.zoom(smooth_field(vol == 1, 1.2), 1.0 / ds_shell, order=1) > 0.4
    sc = mc(smooth_field(shd, 0.8), 0.5, step=2)
    if sc:
        data.append(mesh_trace(sc[0], sc[1], ds_shell * VOX_UM, color="#9aa3b2", opacity=0.12, name="shell"))
    for i, lab in enumerate(labels):
        m = mc(smooth_field(vol == lab, 0.6), 0.4, step=2)
        if m is None:
            continue
        v, f = m
        data.append(mesh_trace(v, f, VOX_UM, color=PALETTE[i % len(PALETTE)], opacity=1.0,
                               name=f"chamber {lab - 1}", showlegend=True))
    fig = go.Figure(data)
    fig.update_layout(**layout("3 · Whole foram — pores coloured by chamber"))
    fig.update_layout(showlegend=True, legend=dict(font=dict(color="#cdd3df"), itemsizing="constant"))
    return fig


def scene_pores_thickness(vol, ds_shell=3):
    """Whole foram: all pores coloured by their own local thickness."""
    print("[pores] local thickness on all pores ...")
    pores = vol >= 2
    plt_ = local_thickness(pores)
    m = mc(smooth_field(pores, 0.9), 0.4, step=2)
    pv, pf = m
    plt_v = sample_at(plt_, pv) * VOX_UM
    cmax = float(np.percentile(plt_v[plt_v > 0], 98))
    data = []
    shd = ndi.zoom(smooth_field(vol == 1, 1.2), 1.0 / ds_shell, order=1) > 0.4
    sc = mc(smooth_field(shd, 0.8), 0.5, step=2)
    if sc:
        data.append(mesh_trace(sc[0], sc[1], ds_shell * VOX_UM, color="#9aa3b2", opacity=0.12, name="shell"))
    data.append(mesh_trace(pv, pf, VOX_UM, intensity=plt_v, colorscale="Hot", cmin=0, cmax=cmax,
                           name="pores", colorbar_title="pore<br>thickness (µm)"))
    fig = go.Figure(data)
    fig.update_layout(**layout("4 · Whole foram — all pores, coloured by local thickness"))
    return fig


def scene_chamber_in_shell(vol, chamber, ds_shell=3):
    print(f"[chamber] highlighting chamber label {chamber} ...")
    pores = vol == chamber
    n = int(pores.sum())
    plt_ = local_thickness(pores)
    m = mc(smooth_field(pores, 0.6), 0.4, step=1)
    data = []
    shd = ndi.zoom(smooth_field(vol == 1, 1.2), 1.0 / ds_shell, order=1) > 0.4
    sc = mc(smooth_field(shd, 0.8), 0.5, step=2)
    if sc:
        data.append(mesh_trace(sc[0], sc[1], ds_shell * VOX_UM, color="#9aa3b2", opacity=0.12, name="shell"))
    title_extra = ""
    if m:
        pv, pf = m
        plt_v = sample_at(plt_, pv) * VOX_UM
        cmax = float(np.percentile(plt_v[plt_v > 0], 98)) if (plt_v > 0).any() else 1
        data.append(mesh_trace(pv, pf, VOX_UM, intensity=plt_v, colorscale="Hot", cmin=0, cmax=cmax,
                               name=f"chamber {chamber}", colorbar_title="pore<br>thickness (µm)"))
        title_extra = f" ({n} pore voxels)"
    fig = go.Figure(data)
    fig.update_layout(**layout(f"5 · Transparent shell + one chamber (label {chamber}){title_extra}"))
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default="MOM_12_01")
    ap.add_argument("--chamber", type=int, default=None,
                    help="data label to highlight in scene 5 (default: 2nd-from-last via chamber_ordering)")
    ap.add_argument("--from-last", type=int, default=2,
                    help="scene 5 chamber = N-th counting back from the last/outermost (default 2)")
    ap.add_argument("--cut-axis", type=int, default=0,
                    help="axis to cut the shell (0 = coiling axis -> equatorial section; default 0)")
    args = ap.parse_args()
    global VOX_UM
    VOX_UM = vox_um_for(args.sample)
    print(f"[vox] {args.sample}: voxel = {VOX_UM*1000:.0f} nm = {VOX_UM} µm")
    vol = np.load(os.path.join(VOL_DIR, f"{args.sample}.npy"))
    labels = sorted(int(l) for l in np.unique(vol) if l >= 2)
    chamber = args.chamber if args.chamber is not None else nth_from_last_label(args.sample, args.from_last)
    if chamber is None or chamber not in labels:
        chamber = labels[1] if len(labels) > 1 else labels[0]
    print(f"[load] {args.sample} {vol.shape}  chambers {labels[0]}..{labels[-1]}  "
          f"highlight={chamber} ({args.from_last}-from-last)")

    figs = [
        ("shell_cut", scene_shell_cut(vol, axis=args.cut_axis)),
        ("shell_tr", scene_shell_transparent(vol)),
        ("pores_all", scene_pores_whole(vol)),
        ("pores_lt", scene_pores_thickness(vol)),
        ("chamber", scene_chamber_in_shell(vol, chamber)),
    ]
    caps = {
        "shell_cut": "<b>1 · Shell cut through the middle.</b> The cut face is coloured by wall local "
                     "thickness (µm), so you read the wall thickness directly in cross-section. Hot = thick.",
        "shell_tr": "<b>2 · Whole shell, semi-transparent</b>, coloured by wall local thickness — front and "
                    "back walls both visible.",
        "pores_all": "<b>3 · All pores on the whole foram, coloured by chamber</b> (one colour per chamber, "
                     "as in the manual-correction editor), inside a faint transparent shell. Toggle chambers in the legend.",
        "pores_lt": "<b>4 · All pores on the whole foram, coloured by local thickness</b> (pore-channel width, µm), "
                    "inside a faint transparent shell.",
        "chamber": f"<b>5 · Transparent whole shell + one chamber's pores</b> (data label "
                   f"{chamber}, picked from chamber_ordering). Shows where that chamber sits in the intact shell.",
    }

    blocks = []
    for i, (k, fig) in enumerate(figs):
        blocks.append(fig.to_html(full_html=False, include_plotlyjs=(i == 0), div_id=k, default_height="640px"))

    config.ensure_output_dirs()
    out = os.path.join(config.OUT, "thickness_3d_report.html")
    cards = "\n".join(f'<div class="card">{blocks[i]}<div class="cap">{caps[k]}</div></div>'
                      for i, (k, _) in enumerate(figs))
    with open(out, "w") as fh:
        fh.write(f"""<!doctype html><html><head><meta charset="utf-8"/>
<title>3-D local thickness — {args.sample}</title>
<style>
body{{margin:0;background:#0f1117;color:#e8eaf0;font:16px/1.6 -apple-system,Segoe UI,Roboto,Arial,sans-serif}}
header{{padding:34px 6vw 16px;background:linear-gradient(135deg,#10231f,#15233a);border-bottom:1px solid #2a3140}}
header h1{{margin:0 0 6px;font-size:25px}} header p{{margin:2px 0;color:#9aa3b2}}
main{{max-width:1200px;margin:0 auto;padding:16px 4vw 60px}}
.card{{background:#1b1f2a;border:1px solid #2a3140;border-radius:14px;padding:10px;margin:18px 0}}
.cap{{color:#cdd3df;font-size:14px;padding:6px 12px 12px}}
.note{{background:#10231f;border-left:4px solid #2aa198;padding:12px 16px;border-radius:8px;color:#dceee9;font-size:14px;margin-top:18px}}
b{{color:#fff}}
</style></head><body>
<header><h1>3-D local thickness &mdash; {args.sample}</h1>
<p>Manually-corrected segmentation &middot; surfaces Gaussian-smoothed &middot; drag to rotate, scroll to zoom</p></header>
<main>
{cards}
<p class="note">Local thickness = Hildebrand definition (largest inscribed sphere), SciPy distance transforms;
voxel = {VOX_UM:.3f} µm ({VOX_UM*1000:.0f} nm, read per-sample from the Ni info table). Shell rendered at
2&ndash;3&times; downsampling for interactivity; pores at full resolution.</p>
</main></body></html>""")
    print("[done]", out, f"({os.path.getsize(out)//1024} KB)")


if __name__ == "__main__":
    main()
