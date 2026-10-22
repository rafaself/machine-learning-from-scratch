"""Execute demonstrations independently of learner answers and reference code."""

import ast
import copy
import os
import re
from pathlib import Path

import nbformat
import pytest
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = sorted((ROOT / "notebooks").glob("*/*.ipynb"))


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda path: path.stem)
def test_demonstrations_run_in_fresh_kernel(path, tmp_path):
    original = path.read_bytes()
    notebook = nbformat.reads(original.decode(), as_version=4)
    demonstrations = []
    reference_functions = set()
    for cell in notebook.cells:
        role = cell.metadata.get("curriculum", {}).get("role")
        if cell.cell_type == "code":
            assert role in {"demonstration", "answer"}, cell.id
            if role == "demonstration":
                demonstrations.append(copy.deepcopy(cell))
        elif role == "solution":
            for source in re.findall(
                r"(?ms)^~~~python[ \t]*\n(.*?)^~~~[ \t]*$", cell.source
            ):
                reference_functions.update(
                    node.name
                    for node in ast.parse(source).body
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                )
    assert demonstrations, "Each chapter must provide executable demonstrations"

    # Only the in-memory copy executes. Student answers are never evaluated,
    # and existing notebook outputs are never cleared or rewritten on disk.
    executable = nbformat.v4.new_notebook(
        cells=demonstrations,
        metadata=copy.deepcopy(notebook.metadata),
    )
    executable.cells.append(
        nbformat.v4.new_code_cell(
            f"assert not (set({sorted(reference_functions)!r}) & set(globals())), "
            "'A demonstration defines a reference implementation'"
        )
    )
    NotebookClient(
        executable,
        kernel_name="python3",
        timeout=90,
        resources={"metadata": {"path": str(tmp_path)}},
    ).execute(
        env={
            **os.environ,
            "MPLBACKEND": "module://matplotlib_inline.backend_inline",
            "MPLCONFIGDIR": str(tmp_path / "matplotlib"),
        }
    )
    assert any(
        "image/png" in output.get("data", {})
        for cell in executable.cells
        for output in cell.get("outputs", [])
    ), "The chapter's visual demonstrations must render"
    assert path.read_bytes() == original, "Execution checks must not rewrite notebooks"
