"""Registry access shared by structural, execution, and numerical checks."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = json.loads((ROOT / "curriculum.json").read_text())


def published_notebooks():
    return [
        entry["path"]
        for stage in REGISTRY["stages"]
        for entry in [*stage["chapters"], stage["capstone"]]
        if entry["status"] == "ready"
    ]


def notebook_path(name):
    matches = [ROOT / path for path in published_notebooks() if Path(path).stem == name]
    if len(matches) != 1:
        raise ValueError(f"Expected one published notebook named {name!r}, found {len(matches)}")
    return matches[0]
