"""Dataset validation. Raises DataValidationError (exit 1) on failure."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from training.common import DataValidationError, load_params


def validate_frame(df: pd.DataFrame, params: dict) -> dict:
    d = params["data"]
    errors: list[str] = []
    expected = list(d["features"]) + [d["label"]]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        errors.append(f"column(s) {missing} missing")
    else:
        if len(df) < d["min_rows"]:
            errors.append(f"row count {len(df)} < minimum {d['min_rows']}")
        for c in d["features"]:
            if not pd.api.types.is_numeric_dtype(df[c]):
                errors.append(f"column '{c}' is not numeric")
        for c in expected:
            ratio = float(df[c].isna().mean())
            if ratio > d["max_null_ratio"]:
                errors.append(f"column '{c}' null ratio {ratio:.2f} > {d['max_null_ratio']}")
        labels = set(df[d["label"]].dropna().unique().tolist())
        if not labels <= {0, 1} or len(labels) < 2:
            errors.append(f"label must be binary with both classes, got {sorted(labels)}")
    return {"ok": not errors, "errors": errors, "rows": len(df), "columns": list(df.columns)}


def validate_file(input_path: str | Path, params: dict, report: str | Path | None = None) -> dict:
    df = pd.read_csv(input_path)
    result = validate_frame(df, params)
    if report:
        Path(report).parent.mkdir(parents=True, exist_ok=True)
        Path(report).write_text(json.dumps(result, indent=2), encoding="utf-8")
    if not result["ok"]:
        raise DataValidationError("DataValidationError: " + "; ".join(result["errors"]))
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default=None)
    p.add_argument("--report", default="validation_report.json")
    p.add_argument("--params", default=None)
    a = p.parse_args()
    params = load_params(a.params)
    try:
        r = validate_file(a.input or params["data"]["path"], params, a.report)
    except DataValidationError as e:
        print(e, file=sys.stderr)
        sys.exit(1)
    print(f"validation ok: {r['rows']} rows")


if __name__ == "__main__":
    main()
