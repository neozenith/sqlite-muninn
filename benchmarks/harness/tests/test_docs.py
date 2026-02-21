"""Tests for docs integration: mkdocs.yml nav entries and doc_page_context."""

from pathlib import Path

import pytest

from benchmarks.harness.analysis.charts_graph import GRAPH_CHARTS
from benchmarks.harness.analysis.charts_graph_vt import GRAPH_VT_CHARTS
from benchmarks.harness.analysis.charts_kg import KG_CHARTS
from benchmarks.harness.analysis.charts_vss import VSS_CHARTS
from benchmarks.harness.registry import DOC_PAGES, doc_page_context

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"

ALL_PAGE_SLUGS = ["vss", "graph", "graph_vt", "kg"]

CHART_MAP = {
    "vss": VSS_CHARTS,
    "graph": GRAPH_CHARTS,
    "graph_vt": GRAPH_VT_CHARTS,
    "kg": KG_CHARTS,
}


# ── mkdocs.yml integration ────────────────────────────────────────


def test_mkdocs_yml_exists():
    mkdocs_path = PROJECT_ROOT / "mkdocs.yml"
    assert mkdocs_path.exists()


def test_mkdocs_nav_has_benchmarks_section():
    mkdocs_path = PROJECT_ROOT / "mkdocs.yml"
    content = mkdocs_path.read_text(encoding="utf-8")
    # String check avoids yaml.safe_load failure on !!python/name: tags
    assert "Benchmarks:" in content, "mkdocs.yml should have a Benchmarks nav section"


def test_mkdocs_nav_has_refactored():
    """After Phase 8, mkdocs.yml should reference refactored benchmark pages."""
    mkdocs_path = PROJECT_ROOT / "mkdocs.yml"
    content = mkdocs_path.read_text(encoding="utf-8")
    # Check for refactored_output in the content
    assert "refactored_output" in content, "mkdocs.yml should reference refactored_output pages"


# ── DOC_PAGES catalog ─────────────────────────────────────────────


def test_doc_pages_has_all_slugs():
    """DOC_PAGES must have an entry for every page slug."""
    for slug in ALL_PAGE_SLUGS:
        assert slug in DOC_PAGES, f"Missing DOC_PAGES entry: {slug}"


@pytest.mark.parametrize("slug", ALL_PAGE_SLUGS)
def test_doc_pages_required_fields(slug):
    """Each DOC_PAGES entry must have title, description, and tables."""
    page = DOC_PAGES[slug]
    assert "title" in page
    assert "description" in page
    assert isinstance(page["title"], str)
    assert isinstance(page["description"], str)
    assert len(page["title"]) > 0
    assert len(page["description"]) > 0


@pytest.mark.parametrize("slug", ALL_PAGE_SLUGS)
def test_doc_pages_tables_structure(slug):
    """Top-level tables must have title, columns, and rows."""
    page = DOC_PAGES[slug]
    tables = page.get("tables", [])
    for table in tables:
        assert "title" in table, f"Table missing title in {slug}"
        assert "columns" in table, f"Table missing columns in {slug}"
        assert "rows" in table, f"Table missing rows in {slug}"
        assert len(table["columns"]) > 0
        for row in table["rows"]:
            for col in table["columns"]:
                assert col in row, f"Row missing column '{col}' in table '{table['title']}' of {slug}"


# ── doc_page_context() ────────────────────────────────────────────


@pytest.mark.parametrize("slug", ALL_PAGE_SLUGS)
def test_doc_page_context_returns_required_keys(slug):
    """doc_page_context must return all keys needed by templates."""
    ctx = doc_page_context(slug)
    assert "title" in ctx
    assert "description" in ctx
    assert "charts" in ctx
    assert "charts_dir" in ctx


@pytest.mark.parametrize("slug", ALL_PAGE_SLUGS)
def test_doc_page_context_charts_match_chart_specs(slug):
    """Chart names in context must match ChartSpec names from charts_*.py."""
    ctx = doc_page_context(slug)
    expected_names = {s.name for s in CHART_MAP[slug]}
    actual_names = {c["name"] for c in ctx["charts"]}
    assert actual_names == expected_names, f"Chart name mismatch for {slug}"


@pytest.mark.parametrize("slug", ALL_PAGE_SLUGS)
def test_doc_page_context_charts_have_titles(slug):
    """Each chart entry must have a name and title."""
    ctx = doc_page_context(slug)
    for chart in ctx["charts"]:
        assert "name" in chart
        assert "title" in chart
        assert len(chart["name"]) > 0
        assert len(chart["title"]) > 0


def test_doc_page_context_unknown_slug_raises():
    """doc_page_context should raise KeyError for unknown slugs."""
    with pytest.raises(KeyError):
        doc_page_context("nonexistent")


def test_doc_page_context_charts_dir_is_relative():
    """charts_dir should be a relative path (not absolute)."""
    ctx = doc_page_context("vss")
    assert not Path(ctx["charts_dir"]).is_absolute()


# ── Template rendering ────────────────────────────────────────────


@pytest.mark.parametrize("slug", ALL_PAGE_SLUGS)
def test_template_exists(slug):
    """Each page slug must have a corresponding .md.j2 template."""
    template_path = TEMPLATES_DIR / f"{slug}.md.j2"
    assert template_path.exists(), f"Missing template: {template_path}"


@pytest.mark.parametrize("slug", ALL_PAGE_SLUGS)
def test_template_renders_without_errors(slug):
    """Templates must render without Jinja2 errors given catalog context."""
    import jinja2

    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(TEMPLATES_DIR)),
        autoescape=False,
    )
    template = env.get_template(f"{slug}.md.j2")
    ctx = doc_page_context(slug)
    content = template.render(**ctx)
    assert len(content) > 0


@pytest.mark.parametrize("slug", ALL_PAGE_SLUGS)
def test_rendered_content_has_title(slug):
    """Rendered markdown must start with the page title as an H1."""
    import jinja2

    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(TEMPLATES_DIR)),
        autoescape=False,
    )
    template = env.get_template(f"{slug}.md.j2")
    ctx = doc_page_context(slug)
    content = template.render(**ctx)
    expected_title = f"# {ctx['title']}"
    assert expected_title in content


@pytest.mark.parametrize("slug", ALL_PAGE_SLUGS)
def test_rendered_content_has_chart_includes(slug):
    """Rendered markdown must have plotly chart includes for each chart."""
    import jinja2

    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(TEMPLATES_DIR)),
        autoescape=False,
    )
    template = env.get_template(f"{slug}.md.j2")
    ctx = doc_page_context(slug)
    content = template.render(**ctx)
    for chart in ctx["charts"]:
        assert chart["name"] in content, f"Chart '{chart['name']}' not found in rendered {slug}.md"
    assert "```plotly" in content


@pytest.mark.parametrize("slug", ALL_PAGE_SLUGS)
def test_rendered_tables_have_no_blank_lines_between_rows(slug):
    """Markdown tables must not have blank lines between rows (breaks parsing)."""
    import jinja2

    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(TEMPLATES_DIR)),
        autoescape=False,
    )
    template = env.get_template(f"{slug}.md.j2")
    ctx = doc_page_context(slug)
    content = template.render(**ctx)

    lines = content.split("\n")
    in_table = False
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("|") and "---" in stripped:
            # This is a separator line — we're in a table
            in_table = True
            continue
        if in_table:
            if stripped.startswith("|"):
                continue  # Still in table
            if stripped == "":
                # Blank line ends the table — that's expected after the last row
                in_table = False
                continue
            # Non-blank, non-pipe line while in_table: table ended
            in_table = False
