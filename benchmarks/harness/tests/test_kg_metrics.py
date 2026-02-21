"""Tests for KG shared evaluation metrics."""

import pytest

from benchmarks.harness.treatments.kg_metrics import bcubed_f1, entity_micro_f1, triple_f1


class TestEntityMicroF1:
    def test_perfect_match(self):
        gold = [(0, 5, "PER"), (10, 15, "ORG")]
        result = entity_micro_f1(gold, gold)
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0
        assert result["f1"] == 1.0

    def test_no_match(self):
        pred = [(0, 5, "PER")]
        gold = [(10, 15, "ORG")]
        result = entity_micro_f1(pred, gold)
        assert result["precision"] == 0.0
        assert result["recall"] == 0.0
        assert result["f1"] == 0.0

    def test_partial_match(self):
        pred = [(0, 5, "PER"), (10, 15, "ORG"), (20, 25, "LOC")]
        gold = [(0, 5, "PER"), (10, 15, "ORG"), (30, 35, "EVT")]
        result = entity_micro_f1(pred, gold)
        # TP=2, FP=1, FN=1
        assert result["precision"] == pytest.approx(2 / 3, abs=0.001)
        assert result["recall"] == pytest.approx(2 / 3, abs=0.001)
        assert 0 < result["f1"] < 1

    def test_empty_both(self):
        result = entity_micro_f1([], [])
        assert result["f1"] == 0.0

    def test_empty_predicted(self):
        result = entity_micro_f1([], [(0, 5, "PER")])
        assert result["precision"] == 0.0
        assert result["recall"] == 0.0
        assert result["f1"] == 0.0

    def test_empty_gold(self):
        result = entity_micro_f1([(0, 5, "PER")], [])
        assert result["precision"] == 0.0
        assert result["recall"] == 0.0
        assert result["f1"] == 0.0

    def test_high_precision_low_recall(self):
        pred = [(0, 5, "PER")]
        gold = [(0, 5, "PER"), (10, 15, "ORG"), (20, 25, "LOC")]
        result = entity_micro_f1(pred, gold)
        assert result["precision"] == 1.0
        assert result["recall"] == pytest.approx(1 / 3, abs=0.001)


class TestTripleF1:
    def test_perfect_match(self):
        gold = [("Alice", "knows", "Bob"), ("Bob", "works_at", "ACME")]
        result = triple_f1(gold, gold)
        assert result["f1"] == 1.0

    def test_no_match(self):
        pred = [("Alice", "knows", "Bob")]
        gold = [("Charlie", "knows", "Diana")]
        result = triple_f1(pred, gold)
        assert result["f1"] == 0.0

    def test_partial_match(self):
        pred = [("A", "r1", "B"), ("C", "r2", "D")]
        gold = [("A", "r1", "B"), ("E", "r3", "F")]
        result = triple_f1(pred, gold)
        assert result["precision"] == 0.5
        assert result["recall"] == 0.5

    def test_empty_both(self):
        result = triple_f1([], [])
        assert result["f1"] == 0.0

    def test_empty_predicted(self):
        result = triple_f1([], [("A", "r", "B")])
        assert result["f1"] == 0.0

    def test_empty_gold(self):
        result = triple_f1([("A", "r", "B")], [])
        assert result["f1"] == 0.0


class TestBCubedF1:
    def test_perfect_clusters(self):
        pred = {"a": 0, "b": 0, "c": 1, "d": 1}
        gold = {"a": 0, "b": 0, "c": 1, "d": 1}
        result = bcubed_f1(pred, gold)
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0
        assert result["f1"] == 1.0

    def test_all_singletons_vs_one_cluster(self):
        pred = {"a": 0, "b": 1, "c": 2}
        gold = {"a": 0, "b": 0, "c": 0}
        result = bcubed_f1(pred, gold)
        # Each element is its own cluster in pred → precision=1.0
        assert result["precision"] == 1.0
        # But recall is low since gold has them all together
        assert result["recall"] == pytest.approx(1 / 3, abs=0.001)

    def test_one_cluster_vs_singletons(self):
        pred = {"a": 0, "b": 0, "c": 0}
        gold = {"a": 0, "b": 1, "c": 2}
        result = bcubed_f1(pred, gold)
        # Recall is 1.0 (each gold cluster member found in pred cluster)
        assert result["recall"] == 1.0
        # Precision is low
        assert result["precision"] == pytest.approx(1 / 3, abs=0.001)

    def test_empty_inputs(self):
        result = bcubed_f1({}, {})
        assert result["f1"] == 0.0

    def test_no_overlap(self):
        pred = {"a": 0, "b": 1}
        gold = {"c": 0, "d": 1}
        result = bcubed_f1(pred, gold)
        assert result["f1"] == 0.0
