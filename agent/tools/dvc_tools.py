"""DVC tools. Subprocess use is restricted to an allowlist of read-only operations."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

from agent.config import get_settings
from agent.registry import ToolRejected, tool
from agent.schemas import (
    DvcCompareInput,
    DvcReproduceInput,
    DvcRunInput,
    NoInput,
)

REV_RE = re.compile(r"^[A-Za-z0-9._/-]{1,100}$")
DEFAULT_PATH = "data/raw/dataset.csv"

# (executable, subcommand tuple) allowed. Anything else is rejected.
ALLOWED = {
    ("dvc", ("diff",)),
    ("dvc", ("status",)),
    ("dvc", ("data", "status")),
    ("dvc", ("list",)),
    ("dvc", ("get",)),
    ("git", ("rev-parse",)),
    ("git", ("log",)),
    ("git", ("tag",)),
    ("git", ("describe",)),
    ("git", ("show",)),
}


def repo() -> Path:
    return Path(get_settings().dvc_repo_path).resolve()


def check_rev(rev: str) -> str:
    if not REV_RE.match(rev) or rev.startswith("-"):
        raise ToolRejected(f"invalid revision '{rev}'")
    return rev


def _allowed(exe: str, sub: tuple[str, ...]) -> bool:
    return any(exe == e and sub[: len(s)] == s for e, s in ALLOWED)


def run_cmd(exe: str, *args: str, timeout: float = 30, check: bool = True) -> str:
    """Run an allowlisted dvc/git command (shell=False). Returns stdout."""
    if not _allowed(exe, tuple(args)):
        raise ToolRejected(f"command '{exe} {' '.join(args[:2])}' is not on the allowlist")
    cmd = [sys.executable, "-m", "dvc", *args] if exe == "dvc" else ["git", *args]
    try:
        r = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=repo(),
            timeout=timeout,
            shell=False,
            env=_env(),
        )
    except subprocess.TimeoutExpired as e:
        raise TimeoutError(f"{exe} {args[0]} timed out after {timeout}s") from e
    if check and r.returncode != 0:
        raise RuntimeError(f"{exe} {' '.join(args)} failed: {r.stderr.strip()[:300]}")
    return r.stdout


def _env() -> dict:
    import os

    e = dict(os.environ)
    e["PYTHONUTF8"] = "1"
    return e


def verify_rev(rev: str) -> str:
    check_rev(rev)
    return run_cmd("git", "rev-parse", "--verify", rev + "^{commit}").strip()


def _parse_dvc_file(text: str) -> dict:
    doc = yaml.safe_load(text) or {}
    out = (doc.get("outs") or [{}])[0]
    return {
        "path": out.get("path"),
        "md5": out.get("md5"),
        "size": out.get("size"),
        "nfiles": out.get("nfiles", 1),
    }


def _default_remote() -> str | None:
    cfg = repo() / ".dvc" / "config"
    if not cfg.exists():
        return None
    name = url = None
    for line in cfg.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s.startswith("remote ="):
            name = s.split("=", 1)[1].strip()
        elif s.startswith("url =") and url is None:
            url = s.split("=", 1)[1].strip()
    return f"{name} ({url})" if name else None


def _dvc_files() -> list[Path]:
    return [
        p
        for p in repo().rglob("*.dvc")
        if not any(part in {".venv", ".git", ".dvc", "tmp"} for part in p.relative_to(repo()).parts)
    ]


@tool(
    "dvc_list_tracked_datasets",
    "List datasets tracked by DVC (path, md5, size, remote).",
    NoInput,
    module="dvc",
)
def list_tracked_datasets(_: NoInput) -> list[dict]:
    remote = _default_remote()
    out = []
    for f in _dvc_files():
        info = _parse_dvc_file(f.read_text(encoding="utf-8"))
        info["dvc_file"] = f.relative_to(repo()).as_posix()
        info["path"] = (f.parent.relative_to(repo()) / (info["path"] or "")).as_posix()
        info["remote"] = remote
        out.append(info)
    return out


@tool(
    "dvc_get_current_version",
    "Current dataset version: HEAD commit, data-v* tag, md5 and remote.",
    NoInput,
    module="dvc",
)
def get_current_version(_: NoInput) -> dict:
    dvc_file = DEFAULT_PATH + ".dvc"
    head = run_cmd("git", "rev-parse", "HEAD").strip()
    try:
        tag = run_cmd("git", "describe", "--tags", "--match", "data-v*", "--abbrev=0").strip()
    except RuntimeError:
        tag = "untagged"
    last = run_cmd("git", "log", "-n1", "--format=%H", "--", dvc_file).strip()
    p = repo() / dvc_file
    md5 = _parse_dvc_file(p.read_text(encoding="utf-8"))["md5"] if p.exists() else None
    return {
        "path": DEFAULT_PATH,
        "version_tag": tag,
        "git_commit": head,
        "dataset_git_commit": last,
        "md5": md5,
        "remote": _default_remote(),
    }


@tool(
    "dvc_list_versions",
    "List dataset versions (git tags data-v*) with commit, md5 and date.",
    NoInput,
    module="dvc",
)
def list_versions(_: NoInput) -> list[dict]:
    tags = [t for t in run_cmd("git", "tag", "-l", "data-v*").split() if t]
    out = []
    for t in sorted(tags):
        commit = run_cmd("git", "rev-parse", "--verify", t + "^{commit}").strip()
        date = run_cmd("git", "log", "-n1", "--format=%cI", t).strip()
        md5 = None
        try:
            txt = run_cmd("git", "show", f"{t}:{DEFAULT_PATH}.dvc")
            md5 = _parse_dvc_file(txt)["md5"]
        except RuntimeError:
            pass
        out.append({"tag": t, "commit": commit, "md5": md5, "date": date})
    return out


def read_csv_at_rev(path: str, rev: str):
    import dvc.api
    import pandas as pd

    with dvc.api.open(path, repo=str(repo()), rev=rev) as f:
        return pd.read_csv(f)


@tool(
    "dvc_compare_versions",
    "Compare a dataset between two revisions (tags/commits): file diff, md5s, row and column "
    "differences.",
    DvcCompareInput,
    module="dvc",
)
def compare_versions(inp: DvcCompareInput) -> dict:
    a, b = verify_rev(inp.rev_a), verify_rev(inp.rev_b)
    path = inp.path or DEFAULT_PATH
    diff = json.loads(run_cmd("dvc", "diff", "--json", inp.rev_a, inp.rev_b) or "{}")
    md5s = []
    for rev in (inp.rev_a, inp.rev_b):
        try:
            md5s.append(_parse_dvc_file(run_cmd("git", "show", f"{rev}:{path}.dvc"))["md5"])
        except RuntimeError:
            md5s.append(None)
    df_a, df_b = read_csv_at_rev(path, inp.rev_a), read_csv_at_rev(path, inp.rev_b)
    return {
        "rev_a": inp.rev_a,
        "rev_b": inp.rev_b,
        "commit_a": a,
        "commit_b": b,
        "added": diff.get("added", []),
        "modified": diff.get("modified", []),
        "deleted": diff.get("deleted", []),
        "md5_a": md5s[0],
        "md5_b": md5s[1],
        "rows_a": len(df_a),
        "rows_b": len(df_b),
        "columns_added": [c for c in df_b.columns if c not in df_a.columns],
        "columns_removed": [c for c in df_a.columns if c not in df_b.columns],
    }


@tool(
    "dvc_check_data_changed",
    "Check whether tracked data changed in workspace vs cache vs remote.",
    NoInput,
    module="dvc",
)
def check_data_changed(_: NoInput) -> dict:
    status = json.loads(run_cmd("dvc", "status", "--json") or "{}")
    data_status = json.loads(run_cmd("dvc", "data", "status", "--json") or "{}")
    uncommitted = data_status.get("uncommitted", {})
    changed = bool(status) or any(uncommitted.get(k) for k in ("modified", "added", "deleted"))
    return {"changed": changed, "dvc_status": status, "data_status": data_status}


@tool(
    "dvc_get_dataset_for_run",
    "Get the dataset version recorded (MLflow provenance tags) for a training run id.",
    DvcRunInput,
    module="dvc",
)
def get_dataset_for_run(inp: DvcRunInput) -> dict:
    from agent.tools.mlflow_tools import get_client

    tags = get_client().get_run(inp.run_id).data.tags
    keys = [
        "data.version",
        "data.git_commit",
        "data.dvc_md5",
        "data.dvc_remote",
        "data.path",
        "code.git_commit",
    ]
    out = {"run_id": inp.run_id, **{k: tags.get(k) for k in keys}}
    out["recorded"] = all(tags.get(k) for k in ("data.version", "data.dvc_md5"))
    return out


@tool(
    "dvc_get_lineage",
    "Show the DVC pipeline lineage from dvc.yaml and dvc.lock.",
    NoInput,
    module="dvc",
)
def get_lineage(_: NoInput) -> dict:
    spec = yaml.safe_load((repo() / "dvc.yaml").read_text(encoding="utf-8")) or {}
    lock_p = repo() / "dvc.lock"
    lock = (yaml.safe_load(lock_p.read_text(encoding="utf-8")) or {}) if lock_p.exists() else {}
    stages = []
    for name, st in (spec.get("stages") or {}).items():
        ls = (lock.get("stages") or {}).get(name, {})
        stages.append(
            {
                "name": name,
                "cmd": st.get("cmd"),
                "deps": [{"path": d["path"], "md5": d.get("md5")} for d in ls.get("deps", [])]
                or [{"path": d, "md5": None} for d in st.get("deps", [])],
                "outs": [{"path": o["path"], "md5": o.get("md5")} for o in ls.get("outs", [])]
                or [
                    {"path": str(o if isinstance(o, str) else next(iter(o))), "md5": None}
                    for o in st.get("outs", [])
                ],
            }
        )
    return {"stages": stages}


def _md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


@tool(
    "dvc_reproduce_dataset_version",
    "Give exact commands to reproduce a dataset version and verify its md5 via a safe "
    "scratch-directory dvc get.",
    DvcReproduceInput,
    module="dvc",
    timeout_s=120,
)
def reproduce_dataset_version(inp: DvcReproduceInput) -> dict:
    verify_rev(inp.rev)
    commands = [
        f"git checkout {inp.rev} -- {DEFAULT_PATH}.dvc",
        f"dvc checkout {DEFAULT_PATH}   # or: dvc pull {DEFAULT_PATH}.dvc",
        f"dvc get . {DEFAULT_PATH} --rev {inp.rev} -o <scratch>/dataset.csv",
    ]
    expected = _parse_dvc_file(run_cmd("git", "show", f"{inp.rev}:{DEFAULT_PATH}.dvc"))["md5"]
    out = {
        "rev": inp.rev,
        "commands": commands,
        "md5_expected": expected,
        "verified": False,
        "md5_actual": None,
    }
    if inp.verify:
        with tempfile.TemporaryDirectory(prefix="dvcget_") as tmp:
            dest = Path(tmp) / "dataset.csv"
            run_cmd("dvc", "get", ".", DEFAULT_PATH, "--rev", inp.rev, "-o", str(dest), timeout=90)
            out["md5_actual"] = _md5(dest)
            out["verified"] = out["md5_actual"] == expected
    return out
