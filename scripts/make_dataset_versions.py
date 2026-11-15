"""Deterministically build dataset versions v1.0, v1.1, v2.0 and a corrupted variant."""

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer

SEED = 42
FEATURES = ["mean radius", "mean texture", "mean perimeter", "mean area", "mean smoothness"]


def base_frame() -> pd.DataFrame:
    ds = load_breast_cancer(as_frame=True)
    df = ds.frame[FEATURES + ["target"]].rename(columns={"target": "label"})
    df.columns = [c.replace(" ", "_") for c in df.columns]
    return df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)


def build(version: str) -> pd.DataFrame:
    df = base_frame()
    cut = int(len(df) * 0.8)
    if version == "v1.0":
        return df.iloc[:cut].reset_index(drop=True)
    if version == "v1.1":
        return df
    if version == "v2.0":
        out = df.copy()
        out["area_per_radius"] = (out["mean_area"] / out["mean_radius"]).round(6)
        return out
    if version == "corrupted":
        out = df.copy()
        rng = np.random.default_rng(SEED)
        idx = rng.choice(len(out), size=int(len(out) * 0.35), replace=False)
        out.loc[idx, "mean_radius"] = np.nan
        return out.drop(columns=["label"])
    raise ValueError(version)


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("version", choices=["v1.0", "v1.1", "v2.0", "corrupted"])
    p.add_argument("--out", default="data/raw/dataset.csv")
    a = p.parse_args()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = build(a.version)
    df.to_csv(out, index=False, lineterminator="\n")
    print(f"{a.version}: rows={len(df)} cols={df.shape[1]} md5={md5(out)} -> {out}")


if __name__ == "__main__":
    main()
