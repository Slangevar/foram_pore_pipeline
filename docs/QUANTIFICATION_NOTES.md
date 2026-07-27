# Quantification — What Changed for the Foram Dataset

This note summarises the minimal changes needed to run `quantification.ipynb` on the MOM foram dataset.
The adapted version is `quantification_foram.ipynb`, which is a direct copy of the original with these edits applied.

---

## Data format

Behnaz's original pipeline stored data as:
- **Whole-foram volumes**: 4-D arrays `(x, y, z, 3)` — one channel per class
- **Per-cluster volumes**: separate `.npy` files per chamber, tracked via `cluster_position_mapping.xlsx` and `top5_num_pores.xlsx`

The foram dataset uses a single **3-D integer-labelled** array per foram:
- `0` = background
- `1` = shell
- `2` = largest chamber (by pore count), `3` = second largest, …, `N` = smallest

All 122 volumes are in: `data/final_foram_state/clustered_volume/*.npy`

This format replaces both the 4-D volumes and the per-cluster files + Excel mapping in one file.

---

## Changes required (4 edits total)

### 1. Paths (Cell 5 / Cell 10)

```python
# Original
DATA_FOLDER  = "../Prediction/ClusterInfo"
OUTPUT_EXCEL = "../Prediction/ClusterInfo/results/quantification_results_cluster.xlsx"

# Change to
DATA_FOLDER    = "data/final_foram_state/clustered_volume"
OUTPUT_RESULTS = "data/analysis/quantification"
OUTPUT_EXCEL   = "data/analysis/quantification/quantification_results.csv"
```

---

### 2. `compute_metrics()` — `mode` → `method` (Cell 3)

The installed porespy version uses `method=` instead of `mode=`:

```python
# Original
local_thickness = ps.filters.local_thickness(binary_data, mode='dt')

# Change to
local_thickness = ps.filters.local_thickness(binary_data, method='dt')
```

Also add an empty-mask guard before this line to avoid crashes on empty chambers:

```python
if not binary_data.any():
    return {"Total_Volume": 0, "Num_Pores": 0, ...}  # zero-filled dict
```

---

### 3. Whole-foram metrics — mask extraction (Cell 6)

```python
# Original (4-D format)
pores_metrics = compute_metrics(data[..., 0])
shell_metrics = compute_metrics(1 - data[..., 2])

# Also: shape check was
if data.ndim != 4 or data.shape[-1] != 3: ...

# Change to (3-D integer labels)
pores_metrics = compute_metrics((data >= 2).astype(np.uint8))  # all pore voxels
shell_metrics = compute_metrics((data == 1).astype(np.uint8))  # shell only

# Shape check becomes
if data.ndim != 3: ...
```

---

### 4. Per-chamber metrics — skip shell label (Cell 14)

```python
# Original (skips background only)
class_labels = class_labels[class_labels > 0]

# Change to (also skip label 1 = shell)
class_labels = class_labels[class_labels >= 2]
```

---

## Cells that are no longer needed

The following cells exist in the original to handle the per-cluster file bookkeeping,
which is already encoded in the integer labels. They can be skipped:

| Cells | Purpose in original | Why not needed |
|-------|--------------------|-----------------------------------------|
| 11    | Read `top5_num_pores.xlsx`, build `cluster_position_mapping.xlsx` | Chamber order is encoded in the label value |
| 12–13 | Walk `ClusterInfo/` folder, plot mid-Z slices of per-cluster files | Single file per foram; no per-cluster files |
| 17–18 | Load `cluster_position_mapping.xlsx` for visualisation | Not needed |
| 28–35 | `process_sample_clusters()` — split volumes into per-cluster files | Already done upstream |
