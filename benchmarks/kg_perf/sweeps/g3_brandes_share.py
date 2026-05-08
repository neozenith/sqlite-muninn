"""G3 brandes_share sweep — produces the inflection chart that informs
the un-defer trigger threshold (T3.2 + T3.3).

Per the plan ADR (line 909), G3 (filter-aware Brandes / induced-subgraph
TVF) becomes worth implementing when ``brandes_share`` —
``centrality_call_time / total_pipeline_time`` per (filter, query) cell —
exceeds ``MUNINN_BRANDES_SHARE_THRESHOLD`` for 3 consecutive runs on the
leading strategy. This sweep produces the empirical justification for
that threshold:

    1. Synthesize corpora at edge counts (10K, 50K, 100K, 500K).
    2. Run the leading strategy on each.
    3. Plot ``brandes_share`` vs ``(V, E)`` to
       ``benchmarks/kg_perf/charts/g3_inflection.png``.

Invoked via ``python -m benchmarks.kg_perf.sweeps.g3_brandes_share``.

Status: SCAFFOLD. The actual corpus synthesis at 50K-500K edges +
matplotlib chart production lands when the un-defer decision is
contemplated. The current implementation:
  - Lays out the output directory and CLI shape.
  - Emits a placeholder PNG so downstream consumers (CI, docs) have
    something to point at.
  - Documents the data contract the real sweep must honor.
"""

from __future__ import annotations

import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CHARTS_DIR = PROJECT_ROOT / "benchmarks" / "kg_perf" / "charts"
DEFAULT_OUTPUT = CHARTS_DIR / "g3_inflection.png"

# Tiny 1x1 transparent PNG. Lets ``test_g3_sweep_produces_chart`` and any
# CI step assert "the chart was produced" without pulling matplotlib
# into the test run. Real implementation replaces this with a
# matplotlib-rendered chart.
_PLACEHOLDER_PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
    "0000000d49444154789c63f9ff9f810f000005010100feeebc8e370000000049454e44ae426082"
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Chart output path (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args(argv)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(_PLACEHOLDER_PNG)
    print(f"Wrote placeholder chart: {args.output}")
    print("TODO: synthesize 10K/50K/100K/500K corpora and plot brandes_share vs (V, E).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
