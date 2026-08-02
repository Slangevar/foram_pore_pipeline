# Quantification inputs

This folder is the default location for the inputs listed in
[`../config.py`](../config.py). Nothing here is tracked by git except this
file — supply your own copies, or point the environment variables elsewhere.

| Path | Env var | Contents |
|---|---|---|
| `clustered_volume/*.npy` | `FORAM_VOL_DIR` | One 3-D `uint8` label volume per specimen: `0` background, `1` shell, `>=2` one label per chamber's pores. These are the output of the manual-correction stage — see [`../../docs/DATA_FORMAT.md`](../../docs/DATA_FORMAT.md). |
| `Ni et al info.xlsx` | `FORAM_NI_INFO` | Voxel sizes, with columns `CT file name` and `resolution nm`. |
| `chamber_ordering.xlsx` | `FORAM_ORDER_XLSX` | Whorl ordering of chambers, needed only for the chamber-wise figures. |

Outputs are written to `../output/` by default (`FORAM_QUANT_DIR` and
`FORAM_FIG_DIR`), which is likewise untracked.

The micro-CT volumes themselves are from Ni et al. (2025) and are archived on
Mendeley Data rather than redistributed here; see the citation section of
[`../README.md`](../README.md).
