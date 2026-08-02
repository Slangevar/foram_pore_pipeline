# final_foram_state

The authoritative, analysis-ready foram dataset. 122 volumes.

This folder is the output of `src/post_processing/reindex_clusters.py` applied to
`data/corrected_cluster_state/all122_addback20_k3_clean/`.

---

## Processing history (in order)

1. **Segmentation** — U-Net prediction on raw micro-CT volumes. Output: `data/predicted_volumes/`
2. **Shell outlier removal** — Remove isolated shell voxels far from the main shell body.
3. **Otsu thresholding** — Threshold-based recovery of additional pore voxels missed by the network.
   Output: `data/pores_cleaned/` (values: `{0=bg, 1=shell, 2=pore}`)
4. **t-SNE + HDBSCAN clustering** (k=3 neighbours) — Cluster pore voxels into chambers using
   morphological features. Output: `data/cluster_state/`
5. **Add-back** (threshold=20 voxels) — Small pore components rejected by HDBSCAN are
   re-assigned to the nearest existing cluster if within distance threshold.
   Output: `data/corrected_cluster_state/all122_addback20_k3_clean/`
6. **Reindexing by pore count** — Chamber labels are remapped so that label 2 always refers to
   the chamber with the most pores, label 3 the second most, etc.
   Output: `data/final_foram_state/` ← **this folder**

---

## Volume encoding

All 3-D `.npy` arrays in this folder use a consistent integer label scheme:

| Value | Meaning |
|-------|---------|
| `0`   | Background (air/resin outside the foram) |
| `1`   | Shell material |
| `2`   | Chamber with the **most** pore voxels |
| `3`   | Chamber with the second most pore voxels |
| `N`   | Chamber ranked N-1 by pore count |

The `predicted_cleaned_volume/` folder is the exception — it only uses `{0, 1, 2}` (undifferentiated pores).

---

## Subfolders

### `predicted_cleaned_volume/`
3-D `uint8` numpy arrays (`.npy`), shape `(Z, Y, X)`. Copied from `data/pores_cleaned/`.

Binary segmentation result after outlier removal and Otsu recovery. Pores are **not yet split
into chambers** — all pore voxels share the single label `2`.

| Value | Meaning |
|-------|---------|
| `0`   | Background |
| `1`   | Shell |
| `2`   | All pore voxels (undifferentiated) |

This is the input that the editor's `volume_path` field points to.
**Use this if you need the raw network output without chamber assignments.**

---

### `clustered_volume/`
3-D `uint8` numpy arrays (`.npy`), shape `(Z, Y, X)`.

Chamber-labelled volumes. Each pore voxel is assigned to a chamber (label ≥ 2) based on
t-SNE + HDBSCAN clustering followed by add-back, and then reindexed by descending pore count.
Background (0) and shell (1) are unchanged from `predicted_cleaned_volume/`.

**Use these for downstream quantification and analysis (`quantification/run_all_quantification.py`).**

---

### `editor_state/`
Lightweight `.npz` files used by the Vue 3 cluster editor
(`src/post_processing/cluster_editor_vue.py`).

Each NPZ contains:

| Key | Description |
|-----|-------------|
| `centroids_3d` | (P, 3) centroid coordinate for each pore component |
| `labels` | (P,) chamber assignment per pore (≥2 = chamber, 0 = deleted) |
| `labeled_pores` | Full 3-D volume with each pore component uniquely labelled |
| `shell_mask` | Binary shell mask |
| `volume_path` | Path to the corresponding `predicted_cleaned_volume/` file |
| `pore_voxels` | Sampled pore voxel coordinates for 3-D rendering |
| `pore_voxels_owner` | Pore index for each sampled voxel |
| `chamber_voxels` | Sampled shell voxels for context rendering in the editor |

**Use these to open a volume in the cluster editor for manual inspection or correction.**
Note: `volume_path` points to `predicted_cleaned_volume/` (the `{0,1,2}` binary segmentation),
not `clustered_volume/`. The editor reconstructs chamber assignments from `labels`.

When saving from the editor, outputs are written to a `manual_review/` subfolder
(separate from this folder), named to avoid confusion with the pipeline-generated outputs.

---

### `summary_npz/`
Machine-readable `.npz` files recording the add-back run statistics per volume:
source/corrected paths, total pore count, number of changed pores, chambers before/after,
and edit timestamp.

### `summary_tsv/`
Human-readable `.tsv` files with per-pore add-back details: pore ID, source component,
voxel count, assigned chamber label, centroid coordinates, and nearest-neighbour distances.

### `addback_manifest.tsv`
One row per volume summarising the add-back run parameters (threshold, k, connectivity, etc.).

### `reindex_summary.csv`
One row per volume recording whether reindexing changed any labels (`remapped=True/False`)
and the final number of chambers.
