"""Rebuild the local DVC remote (dvc-storage/) in a fresh clone.

The remote directory is git-ignored, so a fresh clone has the data-v* tags and .dvc files but
no data. The dataset versions are generated deterministically, so this script regenerates each
tagged version, checks its md5 against the committed .dvc file and pushes it to the remote.
"""

import subprocess
import sys

VERSIONS = {"data-v1.0": "v1.0", "data-v1.1": "v1.1", "data-v2.0": "v2.0"}
DVC_FILE = "data/raw/dataset.csv.dvc"


def run(*cmd: str) -> str:
    return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout


def md5_of_dvc_file() -> str:
    for line in open(DVC_FILE, encoding="utf-8"):
        if "md5:" in line:
            return line.split("md5:")[1].strip()
    raise SystemExit(f"no md5 in {DVC_FILE}")


def main() -> None:
    start = run("git", "rev-parse", "--abbrev-ref", "HEAD").strip()
    if run("git", "status", "--porcelain", "--untracked-files=no").strip():
        raise SystemExit("working tree not clean; commit or stash first")
    try:
        for tag, ver in VERSIONS.items():
            run("git", "checkout", "-q", tag)
            expected = md5_of_dvc_file()
            run(sys.executable, "-m", "scripts.make_dataset_versions", ver)
            run("dvc", "add", "data/raw/dataset.csv")
            got = md5_of_dvc_file()
            if got != expected:
                raise SystemExit(f"{tag}: regenerated md5 {got} != committed {expected}")
            run("dvc", "push")
            run("git", "checkout", "-q", "--", ".")
            print(f"{tag}: md5 {got} restored to remote")
    finally:
        run("git", "checkout", "-q", start)


if __name__ == "__main__":
    main()
