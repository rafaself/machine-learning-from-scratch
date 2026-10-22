"""Structural contracts for reference notebooks and completed study branches."""
import re
from html.parser import HTMLParser
from pathlib import Path

import nbformat
import pytest
from curriculum_support import REGISTRY, ROOT, published_notebooks

NOTEBOOKS = published_notebooks()
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
    path = ROOT / request.param
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
    metadata = notebook.metadata.curriculum
    chapter = metadata.get("exercise_prefix", metadata.get("lesson_number"))
    assert chapter is not None
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


class DisclosureParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.details = []
        self.summaries = 0
        self.closed = 0

    def handle_starttag(self, tag, attrs):
        if tag == "details":
            self.details.append(dict(attrs))
        elif tag == "summary":
            self.summaries += 1

    def handle_endtag(self, tag):
        if tag == "details":
            self.closed += 1


def test_answers_have_adjacent_hint_and_solution(lesson):
    _, notebook = lesson
    reference_ids = {"hint": [], "solution": []}
    answer_ids = []
    for index, cell in enumerate(notebook.cells):
        metadata = cell.metadata.get("curriculum", {})
        role = metadata.get("role")
        if role in reference_ids:
            reference_ids[role].append(metadata["exercise_id"])
        if role != "answer":
            continue
        exercise_id = metadata["exercise_id"]
        answer_ids.append(exercise_id)
        assert index + 2 < len(notebook.cells)
        for offset, expected_role in ((1, "hint"), (2, "solution")):
            reference = notebook.cells[index + offset]
            assert reference.cell_type == "markdown"
            assert reference.metadata.curriculum.role == expected_role
            assert reference.metadata.curriculum.exercise_id == exercise_id
            assert f"reference-{expected_role}" in reference.metadata.get("tags", [])
    assert reference_ids["hint"] == reference_ids["solution"] == answer_ids
    assert notebook.metadata.curriculum.format_version == 2
    assert notebook.metadata.curriculum.reference_solutions == "collapsed-markdown"


def test_references_are_closed_disclosures(lesson):
    _, notebook = lesson
    for cell in notebook.cells:
        role = cell.metadata.get("curriculum", {}).get("role")
        if role not in {"hint", "solution"}:
            continue
        parser = DisclosureParser()
        parser.feed(cell.source)
        assert len(parser.details) == parser.summaries == parser.closed == 1
        assert "open" not in parser.details[0]
        assert cell.source.startswith("<details>\n<summary>")
        assert cell.source.rstrip().endswith("</details>")
        assert f"Exercise {cell.metadata.curriculum.exercise_id}" in cell.source
        expected_label = "Show hint" if role == "hint" else "Show worked solution"
        assert expected_label in cell.source


def test_reference_python_is_displayed_and_compiles(lesson):
    path, notebook = lesson
    exercise_kinds = {
        cell.metadata.curriculum.exercise_id: cell.metadata.curriculum.exercise_type
        for cell in notebook.cells
        if cell.metadata.get("curriculum", {}).get("role") == "exercise"
    }
    for cell in notebook.cells:
        role = cell.metadata.get("curriculum", {}).get("role")
        if role not in {"hint", "solution"}:
            continue
        assert cell.cell_type == "markdown"
        assert "execution_count" not in cell and "outputs" not in cell
        blocks = re.findall(
            r"(?ms)^([~\x60]{3})python[ \t]*\n(.*?)^\1[ \t]*$", cell.source
        )
        for _, source in blocks:
            compile(source, f"{path.name}:{cell.id}", "exec")
        if role == "solution" and exercise_kinds[cell.metadata.curriculum.exercise_id] in {"programming", "experimental"}:
            assert blocks, f"missing reference code: {cell.id}"


def test_published_inventory_matches_registry():
    discovered = {
        str(path.relative_to(ROOT))
        for directory in ("notebooks", "projects")
        for path in (ROOT / directory).rglob("*.ipynb")
        if ".ipynb_checkpoints" not in path.parts
    }
    assert set(NOTEBOOKS) == discovered


def test_registry_readiness_paths_and_prerequisites():
    assert REGISTRY["version"] == 1
    stages = REGISTRY["stages"]
    assert [stage["id"] for stage in stages] == [f"{i:02}" for i in range(9)]
    seen, paths = set(), []
    for stage in stages:
        assert set(stage["prerequisites"]) <= seen
        seen.add(stage["id"])
        entries = [*stage["chapters"], stage["capstone"]]
        statuses = [entry["status"] for entry in entries]
        expected = "ready" if all(s == "ready" for s in statuses) else (
            "partial" if "ready" in statuses else "planned"
        )
        assert stage["status"] == expected
        for entry in entries:
            assert entry["status"] in {"ready", "planned"}
            path = ROOT / entry["path"]
            assert path.resolve().is_relative_to(ROOT)
            assert path.suffix == ".ipynb"
            assert path.is_file() == (entry["status"] == "ready")
            paths.append(entry["path"])
    assert len(paths) == len(set(paths))


def test_exercise_identifiers_are_globally_unique():
    identifiers = []
    for path in NOTEBOOKS:
        notebook = nbformat.read(ROOT / path, as_version=4)
        identifiers.extend(
            cell.metadata.curriculum.exercise_id
            for cell in notebook.cells
            if cell.metadata.get("curriculum", {}).get("role") == "exercise"
        )
    assert len(identifiers) == len(set(identifiers))


def test_stage_guides_cover_mastery_projects_and_compute():
    headings = (
        "## Goal and prerequisites", "## Ordered units",
        "## Implementation requirements", "## Mastery gate",
        "## Capstone", "## Compute and reproducibility", "## Readings",
    )
    for stage in REGISTRY["stages"]:
        guide = (ROOT / stage["directory"] / "README.md").read_text()
        assert f"**Availability:** {stage['status']}." in guide
        assert all(heading in guide for heading in headings)
        assert all(word in guide for word in ("Explain", "Implement", "Verify", "Evaluate"))


def test_documentation_local_links_resolve():
    from urllib.parse import unquote, urlsplit

    for path in ROOT.rglob("*.md"):
        if any(part.startswith(".") for part in path.relative_to(ROOT).parts):
            continue
        for destination in re.findall(r"\]\(([^)]+)\)", path.read_text()):
            url = urlsplit(destination)
            if url.scheme or not url.path:
                continue
            assert (path.parent / unquote(url.path)).resolve().exists(), (path, destination)
