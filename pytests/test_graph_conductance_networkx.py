"""
Cross-check graph_conductance against networkx on random graphs.

networkx.conductance(G, S) = cut_size(S, T) / min(volume(S), volume(T)) with
T = V \\ S, which is the same definition muninn implements. Weighted graphs
use the edge weight for cut and volume in both libraries.
"""

import random

import networkx as nx
import pytest

SEEDS = [1, 2, 3, 7, 42]


def _load_random_graph(conn, seed, n=40, p=0.12, weighted=False):
    """Random undirected G(n, p) graph stored once per edge (u < v), no self loops."""
    rng = random.Random(seed)
    G = nx.gnp_random_graph(n, p, seed=seed)
    conn.execute("CREATE TABLE edges (src TEXT, dst TEXT, weight REAL)")
    for u, v in G.edges():
        w = round(rng.uniform(0.5, 3.0), 3) if weighted else 1.0
        G[u][v]["weight"] = w
        conn.execute("INSERT INTO edges VALUES (?, ?, ?)", (f"n{u}", f"n{v}", w))
    return G


def _random_partition(G, seed, k=4, coverage=1.0):
    """Assign every node (or a `coverage` fraction of them) to one of k groups."""
    rng = random.Random(seed * 101)
    groups = {}
    for node in G.nodes():
        if rng.random() < coverage:
            groups[node] = rng.randrange(k)
    return groups


def _muninn_phi(conn, weighted):
    weight_clause = " AND weight_col = 'weight'" if weighted else ""
    rows = conn.execute(
        "SELECT group_id, size, internal, cut, vol, phi FROM graph_conductance"
        " WHERE edge_table = 'edges' AND src_col = 'src' AND dst_col = 'dst'"
        f"   AND direction = 'both'{weight_clause}"
        "   AND membership_table = 'membership' AND group_col = 'grp' AND member_col = 'node'"
    ).fetchall()
    return {int(r[0]): r[1:] for r in rows}


def _networkx_reference(G, groups, weighted):
    """Per-group (size, internal, cut, vol, phi) using only networkx primitives."""
    w = "weight" if weighted else None
    out = {}
    for gid in sorted(set(groups.values())):
        S = {n for n, g in groups.items() if g == gid}
        T = set(G.nodes()) - S
        cut = nx.cut_size(G, S, T, weight=w)
        vol = nx.volume(G, S, weight=w)
        internal = (vol - cut) / 2
        phi = nx.conductance(G, S, T, weight=w) if min(vol, nx.volume(G, T, weight=w)) > 0 else 0.0
        out[gid] = (len(S), internal, cut, vol, phi)
    return out


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("weighted", [False, True], ids=["unweighted", "weighted"])
def test_full_partition_matches_networkx(conn, seed, weighted):
    G = _load_random_graph(conn, seed, weighted=weighted)
    groups = _random_partition(G, seed)
    conn.execute("CREATE TABLE membership (node TEXT, grp INTEGER)")
    conn.executemany("INSERT INTO membership VALUES (?, ?)", [(f"n{n}", g) for n, g in groups.items()])

    expected = _networkx_reference(G, groups, weighted)
    actual = _muninn_phi(conn, weighted)

    # groups whose members all have degree 0 are absent from muninn (no edges -> not in graph)
    expected = {g: v for g, v in expected.items() if v[3] > 0}
    assert set(actual) == set(expected)
    for gid, (_size, internal, cut, vol, phi) in expected.items():
        a_size, a_internal, a_cut, a_vol, a_phi = actual[gid]
        # muninn's size counts only nodes present in the edge table
        assert a_size == len([n for n, g in groups.items() if g == gid and G.degree(n) > 0])
        assert a_internal == pytest.approx(internal, abs=1e-9)
        assert a_cut == pytest.approx(cut, abs=1e-9)
        assert a_vol == pytest.approx(vol, abs=1e-9)
        assert a_phi == pytest.approx(phi, abs=1e-9)


@pytest.mark.parametrize("seed", SEEDS)
def test_partial_membership_matches_networkx(conn, seed):
    """Ungrouped nodes: muninn scores each group against the whole graph, as networkx does with T = V - S."""
    G = _load_random_graph(conn, seed)
    groups = _random_partition(G, seed, coverage=0.6)
    conn.execute("CREATE TABLE membership (node TEXT, grp INTEGER)")
    conn.executemany("INSERT INTO membership VALUES (?, ?)", [(f"n{n}", g) for n, g in groups.items()])

    expected = {g: v for g, v in _networkx_reference(G, groups, False).items() if v[3] > 0}
    actual = _muninn_phi(conn, False)

    assert set(actual) == set(expected)
    for gid, (_, internal, cut, vol, phi) in expected.items():
        assert actual[gid][1:] == pytest.approx((internal, cut, vol, phi), abs=1e-9)
