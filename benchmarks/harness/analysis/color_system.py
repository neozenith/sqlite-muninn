"""Fiber-bundle color assignment system.

Hue is fixed per engine-algorithm group.
Saturation/Luminance vary per variant (model/dimension) within a hue.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from benchmarks.harness.analysis.aggregator import ChartSeries

# Base hue per engine-algorithm (HSL hue degrees)
ENGINE_HUES: dict[str, int] = {
    # VSS engines
    "muninn_hnsw": 270,  # purple
    "sqlite_vector_quantize_scan": 175,  # teal
    "sqlite_vector_full_scan": 18,  # warm orange
    "vectorlite_hnsw": 210,  # blue
    "sqlite_vec_brute_force": 130,  # green
    # Graph engines (plain — used when group_fields=["engine"])
    "muninn": 270,  # purple
    "graphqlite": 130,  # green
    # Graph engines (operation-specific — legacy keys)
    "muninn_bfs": 270,
    "muninn_dfs": 270,
    "muninn_shortest_path": 270,
    "muninn_components": 270,
    "muninn_pagerank": 270,
    "muninn_degree": 270,
    "muninn_betweenness": 270,
    "muninn_closeness": 270,
    "muninn_leiden": 270,
    "graphqlite_bfs": 130,
    "graphqlite_dfs": 130,
    "graphqlite_shortest_path": 130,
    "graphqlite_components": 130,
    "graphqlite_pagerank": 130,
    # Graph models (when used as group keys in community charts)
    "erdos_renyi": 30,  # orange
    "barabasi_albert": 200,  # blue
    # Centrality operations (when used as group keys)
    "degree": 270,  # purple
    "betweenness": 330,  # pink
    "closeness": 175,  # teal
    # Graph VT approaches (match legacy adjacency colors)
    "tvf": 0,  # red
    "csr": 30,  # orange
    "csr_full_rebuild": 30,  # orange
    "csr_incremental": 160,  # teal
    "csr_blocked": 270,  # purple
}

# Default hue for unknown engines
DEFAULT_HUE = 0

# Human-readable display labels for series names
DISPLAY_LABELS: dict[str, str] = {
    # VSS engines (group_key = "engine_searchmethod")
    "muninn_hnsw": "muninn-hnsw",
    "sqlite_vector_quantize_scan": "sqlite-vector-quantize",
    "sqlite_vector_full_scan": "sqlite-vector-fullscan",
    "vectorlite_hnsw": "vectorlite-hnsw",
    "sqlite_vec_brute_force": "sqlite-vec-brute",
    # Graph engines
    "muninn": "muninn",
    "graphqlite": "graphqlite",
    # Graph models
    "erdos_renyi": "Erdos-Renyi",
    "barabasi_albert": "Barabasi-Albert",
    # Centrality operations
    "degree": "Degree",
    "betweenness": "Betweenness",
    "closeness": "Closeness",
    # Graph VT approaches
    "tvf": "TVF (no cache)",
    "csr": "CSR",
    "csr_full_rebuild": "CSR full-rebuild",
    "csr_incremental": "CSR incremental",
    "csr_blocked": "CSR blocked-incremental",
}


def assign_color(group_key: str, variant_idx: int = 0, n_variants: int = 1) -> str:
    """Assign an HSL color based on group key and variant index.

    Args:
        group_key: Underscore-joined group field values (e.g., "muninn_hnsw").
        variant_idx: Index of this variant within the group (0-based).
        n_variants: Total number of variants in this group.

    Returns:
        HSL color string (e.g., "hsl(270, 75%, 45%)").
    """
    hue = ENGINE_HUES.get(group_key, DEFAULT_HUE)

    if n_variants <= 1:
        sat, lum = 75, 45
    else:
        t = variant_idx / (n_variants - 1)
        sat = 85 - int(t * 15)  # 85% -> 70%
        lum = 58 - int(t * 23)  # 58% -> 35%

    return f"hsl({hue}, {sat}%, {lum}%)"


def assign_colors(series_list: list[ChartSeries]) -> None:
    """Assign colors to all series based on their group/variant keys.

    Modifies series in place by setting their `color` attribute.
    """
    # Group by group_key to count variants within each group
    groups: dict[str, list[ChartSeries]] = {}
    for s in series_list:
        groups.setdefault(s.group_key, []).append(s)

    for group_key, group_series in groups.items():
        # Sort variants lexicographically for consistent assignment
        group_series.sort(key=lambda s: s.variant_key)
        n_variants = len(group_series)
        for idx, s in enumerate(group_series):
            s.color = assign_color(group_key, variant_idx=idx, n_variants=n_variants)
