# Scripts Organization

This folder contains categorized operational scripts.

- `run_job.sh`: model training
- `predict_job.sh`: batch prediction + HTML visualization
- `clean_job.sh`: LCC outlier cleaning + HTML visualization
- `run_pipeline_batch.sh`: full CPU pipeline for pores cleaning + clustering + editor data
- `run_pipeline_test.sh`: subset test pipeline
- `run_reprepare_only.sh`: regenerate editor state with updated voxel sampling ratio/cap
- `cluster_job.sh`: sample clustering sanity run
- `run_verification.sh`: quick training verification run
- `run_otsu_recovery_analysis.sh`: batch audit for Otsu pore removal and fixed-threshold pore recovery

## `maintenance/`
Utility scripts for environment and assets.

## `analysis/`
Analysis/reporting helpers:
- `calculate_stats.py`: parse outlier-cleaning stats from logs into CSV
- `analyze_otsu_pore_recovery.py`: audit per-volume pores removed by Otsu and estimate recovery using a fixed size threshold (default 20 voxels), exporting both CSV and Markdown summary

## Backward compatibility
