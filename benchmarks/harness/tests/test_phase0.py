"""Phase 0 verification: spec file and introspection artifacts exist."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def test_spec_file_exists():
    spec = PROJECT_ROOT / "docs" / "plans" / "benchmark_harness_refactor.md"
    assert spec.exists(), f"Spec file not found: {spec}"
    content = spec.read_text(encoding="utf-8")
    assert "Treatment" in content, "Spec should mention the Treatment pattern"


def test_introspection_artifacts_exist():
    base = PROJECT_ROOT / "docs" / "plans" / "benchmarks_refactor"
    assert base.is_dir(), f"Introspection dir not found: {base}"

    work_events = base / "work_events.json"
    assert work_events.exists(), f"Missing: {work_events}"

    user_reqs = base / "original_user_requirement_events.json"
    assert user_reqs.exists(), f"Missing: {user_reqs}"

    requirements = base / "script_requirements_todate.md"
    assert requirements.exists(), f"Missing: {requirements}"
