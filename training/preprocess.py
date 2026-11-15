"""Seeded split and scaling; writes data/processed/."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from training.common import load_params


def preprocess(input_path: str | Path, out_dir: str | Path, params: dict) -> dict:
    d, s = params["data"], params["split"]
    df = pd.read_csv(input_path)
    feats, label = d["features"], d["label"]
    train, test = train_test_split(
        df, test_size=s["test_size"], random_state=s["seed"], stratify=df[label]
    )
    scaler = StandardScaler().fit(train[feats])
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, part in (("train", train), ("test", test)):
        scaled = pd.DataFrame(scaler.transform(part[feats]), columns=feats, index=part.index)
        scaled[label] = part[label].values
        scaled.to_csv(out / f"{name}.csv", index=False)
    joblib.dump(scaler, out / "scaler.joblib")
    return {"train_rows": len(train), "test_rows": len(test)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default=None)
    p.add_argument("--out", default="data/processed")
    a = p.parse_args()
    params = load_params()
    print(preprocess(a.input or params["data"]["path"], a.out, params))


if __name__ == "__main__":
    main()
