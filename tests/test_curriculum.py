"""Structural contracts for reference notebooks and completed study branches."""
import re
from pathlib import Path

import nbformat
import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = (
    "01-foundations/linear-regression.ipynb",
    "01-foundations/gradient-descent.ipynb",
    "01-foundations/logistic-regression.ipynb",
    "02-neural-networks/perceptron.ipynb",
    "02-neural-networks/backpropagation.ipynb",
    "02-neural-networks/multilayer-perceptron.ipynb",
)
SECTIONS = (
    "Introduction and learning objectives",
    "Prerequisites and mathematical foundations",
    "Theoretical development and worked examples",
    "Guided exercises",
    "Implementation challenges",
    "Experiments and visualizations",
    "Interpretation and critical analysis",
    "Review questions and knowledge checks",
    "Summary and next learning steps",
)
KINDS = {"conceptual", "mathematical", "numerical", "programming", "experimental", "critical"}


@pytest.fixture(params=NOTEBOOKS, ids=lambda value: Path(value).stem)
def lesson(request):
    path = ROOT / "notebooks" / request.param
    return path, nbformat.read(path, as_version=4)


def test_notebook_schema_and_python_kernel(lesson):
    _, notebook = lesson
    nbformat.validate(notebook)
    assert notebook.metadata.kernelspec.name == "python3"
    assert notebook.metadata.kernelspec.language == "python"
    ids = [cell.id for cell in notebook.cells]
    assert len(ids) == len(set(ids))


def test_chapter_sections_are_in_order(lesson):
    _, notebook = lesson
    headings = []
    for cell in notebook.cells:
        if cell.cell_type == "markdown":
            headings.extend(re.findall(r"^## ([1-9])\. (.+)$", cell.source, re.MULTILINE))
    assert headings == [(str(index), title) for index, title in enumerate(SECTIONS, 1)]


def test_exercises_have_adjacent_typed_answer_cells(lesson):
    _, notebook = lesson
    exercise_ids = []
    answer_ids = []
    for index, cell in enumerate(notebook.cells):
        metadata = cell.metadata.get("curriculum", {})
        if metadata.get("role") == "answer":
            answer_ids.append(metadata["exercise_id"])
            assert index > 0
            assert notebook.cells[index - 1].metadata.get("curriculum", {}).get("role") == "exercise"
        if metadata.get("role") != "exercise":
            continue
        exercise_id = metadata["exercise_id"]
        exercise_ids.append(exercise_id)
        assert metadata["exercise_type"] in KINDS
        assert cell.cell_type == "markdown"
        assert f"### Exercise {exercise_id} — " in cell.source
        assert "**Expected outcome:**" in cell.source
        assert index + 1 < len(notebook.cells)
        answer = notebook.cells[index + 1]
        answer_metadata = answer.metadata.get("curriculum", {})
        assert answer_metadata.get("role") == "answer"
        assert answer_metadata.get("exercise_id") == exercise_id
        expected_type = "code" if metadata["exercise_type"] in {"programming", "experimental"} else "markdown"
        assert answer.cell_type == expected_type
        assert answer_metadata.get("answer_format") == metadata["answer_format"]
        # Student answers are allowed: content is deliberately not constrained.
    assert exercise_ids
    assert answer_ids == exercise_ids
    chapter = notebook.metadata.curriculum.lesson_number
    assert exercise_ids == [f"{chapter}.{number:02}" for number in range(1, len(exercise_ids) + 1)]


def test_local_notebook_links_resolve(lesson):
    path, notebook = lesson
    for cell in notebook.cells:
        if cell.cell_type != "markdown":
            continue
        for relative in re.findall(r"\]\(([^)]+\.ipynb)\)", cell.source):
            assert (path.parent / relative).resolve().is_file(), relative


def test_all_exercise_types_are_available(lesson):
    _, notebook = lesson
    kinds = {
        cell.metadata["curriculum"]["exercise_type"]
        for cell in notebook.cells
        if cell.metadata.get("curriculum", {}).get("role") == "exercise"
    }
    assert kinds == KINDS


def test_all_planned_topics_exist():
    assert set(NOTEBOOKS) <= {
        str(path.relative_to(ROOT / "notebooks"))
        for path in (ROOT / "notebooks").rglob("*.ipynb")
        if ".ipynb_checkpoints" not in path.parts
    }
