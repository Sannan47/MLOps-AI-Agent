# Datasets

| Path | Tracked by | Versions (git tags) |
|---|---|---|
| data/raw/dataset.csv | data/raw/dataset.csv.dvc | data-v1.0 (455 rows), data-v1.1 (569 rows), data-v2.0 (569 rows, adds `area_per_radius`) |

Source: scikit-learn Breast Cancer Wisconsin, 5 features + binary `label`. Rebuild with `scripts/make_dataset_versions.py`.
Remote: `localstore` (dvc-storage/, git-ignored). A `corrupted` variant exists only for failure injection and is never tagged.
