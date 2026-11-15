from __future__ import annotations

import json
import subprocess
from unittest.mock import patch

import pandas as pd
import pytest

from agent.registry import registry
from agent.tools import dvc_tools


def test_dvc_rejects_disallowed_command_and_bad_rev():
    with patch("subprocess.run") as run:
        with pytest.raises(Exception, match="allowlist"):
            dvc_tools.run_cmd("dvc", "push")
        with pytest.raises(Exception, match="allowlist"):
            dvc_tools.run_cmd("git", "checkout", "main")
        with pytest.raises(Exception, match="invalid revision"):
            dvc_tools.check_rev("main; rm -rf /")
        with pytest.raises(Exception, match="invalid revision"):
            dvc_tools.check_rev("--upload-pack=x")
        run.assert_not_called()
    res = registry.execute("dvc_compare_versions", {"rev_a": "a b", "rev_b": "x"}, "r")
    assert res.status == "rejected"


def _fake_run(outputs: dict):
    def run(cmd, **kw):
        key = " ".join(cmd[1:] if cmd[0] == "git" else cmd[3:])
        for pat, out in outputs.items():
            if pat in key:
                return subprocess.CompletedProcess(cmd, 0, stdout=out, stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    return run


def test_dvc_compare_versions_parses_diff():
    diff = json.dumps({"added": [], "deleted": [], "modified": [{"path": "data/raw/dataset.csv"}]})
    outputs = {
        "rev-parse --verify": "abc1234\n",
        "diff --json": diff,
        "show": "outs:\n- md5: deadbeef\n  path: dataset.csv\n",
    }
    dfs = {
        "v1": pd.DataFrame({"a": [1, 2], "label": [0, 1]}),
        "v2": pd.DataFrame({"a": [1, 2, 3], "label": [0, 1, 1], "new": [1, 1, 1]}),
    }
    with (
        patch("subprocess.run", _fake_run(outputs)),
        patch.object(dvc_tools, "read_csv_at_rev", lambda path, rev: dfs[rev]),
    ):
        res = registry.execute("dvc_compare_versions", {"rev_a": "v1", "rev_b": "v2"}, "r")
    assert res.status == "ok", res.error
    d = res.data
    assert d["modified"] and d["rows_b"] > d["rows_a"] and d["columns_added"] == ["new"]


def test_dvc_subprocess_timeout_gives_timeout_status():
    def boom(*a, **k):
        raise subprocess.TimeoutExpired("dvc", 1)

    with patch("subprocess.run", boom):
        res = registry.execute("dvc_check_data_changed", {}, "r")
    assert res.status == "timeout"


def test_dvc_list_versions_parses_tags():
    outputs = {
        "tag -l": "data-v1.0\ndata-v1.1\n",
        "rev-parse --verify": "c0ffee1\n",
        "log -n1": "2026-01-01T00:00:00+00:00\n",
        "show": "outs:\n- md5: 1234abcd\n  path: dataset.csv\n",
    }
    with patch("subprocess.run", _fake_run(outputs)):
        res = registry.execute("dvc_list_versions", {}, "r")
    assert [v["tag"] for v in res.data] == ["data-v1.0", "data-v1.1"]
    assert res.data[0]["md5"] == "1234abcd"


def test_dvc_get_dataset_for_run_uses_recorded_tags():
    from types import SimpleNamespace

    tags = {
        "data.version": "data-v1.1",
        "data.git_commit": "abc",
        "data.dvc_md5": "m",
        "data.dvc_remote": "localstore (x)",
        "data.path": "p",
        "code.git_commit": "c",
    }
    fake = SimpleNamespace(get_run=lambda rid: SimpleNamespace(data=SimpleNamespace(tags=tags)))
    with patch("agent.tools.mlflow_tools.get_client", lambda: fake):
        res = registry.execute("dvc_get_dataset_for_run", {"run_id": "r1"}, "r")
    assert res.status == "ok" and res.data["data.version"] == "data-v1.1" and res.data["recorded"]


@pytest.mark.integration
def test_integration_compare_real_repo():
    res = registry.execute(
        "dvc_compare_versions", {"rev_a": "data-v1.0", "rev_b": "data-v1.1"}, "r"
    )
    assert res.status == "ok" and res.data["modified"] and res.data["rows_b"] > res.data["rows_a"]
