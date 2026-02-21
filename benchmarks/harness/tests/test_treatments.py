"""Tests for all treatment categories — 1 smoke test per category.

Verifies that every treatment category has at least 1 registered permutation,
all permutation_ids are unique across all categories, and each treatment
subclass can be instantiated with params_dict() returning a non-empty dict.
"""

from benchmarks.harness.registry import all_permutations


class TestAllTreatments:
    def test_every_category_has_permutations(self):
        """Every expected category should have at least 1 permutation."""
        perms = all_permutations()
        categories = {p.category for p in perms}
        expected = {
            "vss",
            "graph",
            "centrality",
            "community",
            "graph_vt",
            "kg-extract",
            "kg-re",
            "kg-resolve",
            "kg-graphrag",
            "node2vec",
        }
        for cat in expected:
            assert cat in categories, f"No permutations for category: {cat}"

    def test_all_ids_unique_across_categories(self):
        """All permutation IDs must be globally unique."""
        perms = all_permutations()
        ids = [p.permutation_id for p in perms]
        duplicates = [pid for pid in ids if ids.count(pid) > 1]
        assert not duplicates, f"Duplicate permutation IDs: {set(duplicates)}"

    def test_params_dict_nonempty_for_each_category(self):
        """At least one treatment per category should return non-empty params_dict."""
        perms = all_permutations()
        by_category = {}
        for p in perms:
            by_category.setdefault(p.category, []).append(p)

        for cat, cat_perms in by_category.items():
            first = cat_perms[0]
            params = first.params_dict()
            assert isinstance(params, dict), f"params_dict for {cat} should be dict"
            assert len(params) > 0, f"params_dict for {cat}/{first.permutation_id} should be non-empty"

    def test_total_permutation_count_is_reasonable(self):
        """Sanity check: total count should be > 50 (lots of permutations)."""
        perms = all_permutations()
        assert len(perms) > 50, f"Expected > 50 permutations, got {len(perms)}"

    def test_category_counts(self):
        """Report permutation counts per category."""
        perms = all_permutations()
        by_category = {}
        for p in perms:
            by_category.setdefault(p.category, []).append(p)

        # VSS should have the most (5 engines x 3 models x 2 datasets x 6 sizes)
        assert len(by_category.get("vss", [])) >= 10
        # Graph should have multiple (2 engines x 5 ops x several configs)
        assert len(by_category.get("graph", [])) >= 10
        # Each other category should have at least a few
        for cat in [
            "centrality",
            "community",
            "graph_vt",
            "kg-extract",
            "kg-re",
            "kg-resolve",
            "kg-graphrag",
            "node2vec",
        ]:
            assert len(by_category.get(cat, [])) >= 2, f"Category {cat} has too few permutations"
