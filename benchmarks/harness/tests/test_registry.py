"""Tests for the permutation registry."""

import re

from benchmarks.harness.registry import all_permutations, filter_permutations, permutation_status


class TestRegistry:
    def test_returns_non_empty(self):
        perms = all_permutations()
        assert len(perms) > 0, "Registry should return at least one permutation"

    def test_all_permutation_ids_unique(self):
        perms = all_permutations()
        ids = [p.permutation_id for p in perms]
        assert len(ids) == len(set(ids)), f"Duplicate permutation IDs found: {len(ids)} total, {len(set(ids))} unique"

    def test_permutation_ids_are_valid_path_components(self):
        """All permutation IDs should be safe for use as directory names."""
        perms = all_permutations()
        # Allow alphanumeric, hyphens, underscores, dots
        valid_pattern = re.compile(r"^[a-zA-Z0-9._-]+$")
        for p in perms:
            assert valid_pattern.match(p.permutation_id), (
                f"Invalid permutation ID (not filesystem-safe): {p.permutation_id}"
            )

    def test_all_have_category(self):
        perms = all_permutations()
        for p in perms:
            assert p.category, f"Missing category for {p.permutation_id}"

    def test_all_have_label(self):
        perms = all_permutations()
        for p in perms:
            assert p.label, f"Missing label for {p.permutation_id}"

    def test_categories_present(self):
        """All expected categories should have at least one permutation."""
        perms = all_permutations()
        categories = {p.category for p in perms}
        expected = {
            "vss",
            "graph",
            "centrality",
            "community",
            "graph_vt",
            "kg-extract",
            "kg-resolve",
            "kg-graphrag",
            "node2vec",
        }
        for cat in expected:
            assert cat in categories, f"Missing category: {cat}"

    def test_filter_by_category(self):
        vss = filter_permutations(category="vss")
        assert len(vss) > 0
        assert all(p.category == "vss" for p in vss)

    def test_filter_by_nonexistent_category(self):
        result = filter_permutations(category="nonexistent")
        assert result == []

    def test_filter_by_id(self):
        perms = all_permutations()
        first_id = perms[0].permutation_id
        result = filter_permutations(permutation_id=first_id)
        assert len(result) == 1
        assert result[0].permutation_id == first_id


class TestPermutationStatus:
    def test_returns_list_of_dicts(self, tmp_path):
        status = permutation_status(results_dir=tmp_path)
        assert isinstance(status, list)
        assert len(status) > 0
        first = status[0]
        assert "permutation_id" in first
        assert "category" in first
        assert "label" in first
        assert "done" in first

    def test_all_missing_when_empty_dir(self, tmp_path):
        status = permutation_status(results_dir=tmp_path)
        assert all(not s["done"] for s in status)

    def test_detects_done_permutation(self, tmp_path):
        # Create a fake completed permutation
        perms = all_permutations()
        first = perms[0]
        perm_dir = tmp_path / first.permutation_id
        perm_dir.mkdir()
        (perm_dir / "db.sqlite").write_text("fake")

        status = permutation_status(results_dir=tmp_path)
        done_ids = {s["permutation_id"] for s in status if s["done"]}
        assert first.permutation_id in done_ids
