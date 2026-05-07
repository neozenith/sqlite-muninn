# `/kg/er/` Cytoscape Visualization — Architecture

Architecture reference for the entity-resolved knowledge-graph view served at
`/<database_id>/kg/er/` (and `/<database_id>/kg/base/`). Two diagrams: a
narrative overview that fits the page, and a detailed reference that traces
every layer from a user click to a SQLite query.

## What this view does

For each demo database, the page renders the knowledge graph as a
Cytoscape.js canvas. Nodes are entities (or canonical clusters when
`tableId === 'er'`); edges are typed relations; community parents are
compound nodes from the Leiden clustering. The right panel exposes
collapsible sections for filters, selection inspection, server-side graph
parameters, fcose layout constraints, aesthetic styling, and PNG/SVG
export.

---

## Architecture overview

```mermaid
flowchart LR
    user(["User"]):::actor

    subgraph browser["Browser"]
        kg["KGPage<br/>(/kg/:tableId/)"]:::browserPrimary
        cy["Cytoscape + fcose<br/>(layout, selection)"]:::browserPrimary
        panel["Right Panel<br/>(6 collapsible sections)"]:::browserSecondary
        exp["PNG / SVG<br/>download"]:::export
    end

    subgraph backend["Backend"]
        api["/api/databases/:id/kg/:tableId/"]:::apiPrimary
        svc["load_kg_graph()<br/>NetworkX compute"]:::computePrimary
    end

    db[("SQLite<br/>{book}_{model}.db")]:::dataPrimary

    user -->|click, drag| cy
    user -->|edit controls| panel
    panel -->|topN / depth / pin / theme| kg
    kg -->|GET kg payload| api
    api --> svc
    svc -->|read tables| db
    svc -->|JSON KGPayload| kg
    kg -->|elements + stylesheet| cy
    cy -->|cy.png / cy.svg| exp
    exp -->|browser download| user

    classDef actor              fill:#cbd5e1,stroke:#334155,color:#1e293b,stroke-width:1px
    classDef browserPrimary     fill:#2563eb,stroke:#000000,color:#fff,stroke-width:2px
    classDef browserSecondary   fill:#93c5fd,stroke:#1e3a8a,color:#1e293b,stroke-width:1px
    classDef apiPrimary         fill:#7c3aed,stroke:#000000,color:#fff,stroke-width:2px
    classDef computePrimary     fill:#7c3aed,stroke:#000000,color:#fff,stroke-width:2px
    classDef dataPrimary        fill:#0f766e,stroke:#000000,color:#fff,stroke-width:2px
    classDef export             fill:#047857,stroke:#000000,color:#fff,stroke-width:2px

    classDef sgBrowser fill:#dbeafe,stroke:#1e3a8a,color:#1e293b
    classDef sgBackend fill:#ede9fe,stroke:#5b21b6,color:#1e293b
    class browser sgBrowser
    class backend sgBackend
```

**Read-it-as:** the user drives Cytoscape clicks and the right-panel
controls; the panel and KGPage between them own all client state; data
flows once per parameter change from Cytoscape's host page → FastAPI →
the per-database SQLite file → back to Cytoscape as a JSON payload that
becomes elements; PNG/SVG export goes straight from the in-browser
Cytoscape instance to the user's filesystem (no round-trip).

---

## Detailed reference

<details>
<summary>Full pipeline — every component, every read path</summary>

```mermaid
flowchart TB
    user(["User input<br/>(clicks / drags / typing)"]):::actor

    subgraph rp["Right panel (collapsible sections)"]
        s1["Filters<br/>(entity / rel legends)"]:::panelTier
        s2["Selection<br/>(nodes / edges / communities)"]:::panelTier
        s3["Graph filters<br/>(topN, seed, depth, min-deg)"]:::panelTier
        s4["Constraints + JSON<br/>(fcose source-of-truth)"]:::panelTier
        s5["Aesthetics<br/>(size / color / opacity)"]:::panelTier
        s6["Export<br/>(PNG / SVG buttons)"]:::panelTier
    end

    subgraph fe["Browser (KGPage)"]
        st["React state<br/>(data axis, layoutConfig,<br/>selection)"]:::browserSecondary
        eff["Effects<br/>(debounced fetch,<br/>auto-apply layout)"]:::browserSecondary
        kgp["KGPage component"]:::browserPrimary
        cyc["Cytoscape.js core"]:::browserPrimary
        fcose["cytoscape-fcose<br/>(constraints layout)"]:::pluginTier
        svg["cytoscape-svg<br/>(SVG serializer)"]:::pluginTier
        pngOut(["*.png download"]):::export
        svgOut(["*.svg download"]):::export
    end

    subgraph be["Backend (FastAPI)"]
        proxy["Vite proxy<br/>/api/* → :8290"]:::infraSecondary
        rt["GET /api/databases/<br/>:id/kg/:tableId/"]:::apiPrimary
        lg["load_kg_graph()<br/>(server/kg.py)"]:::computePrimary
        lb["_load_base()"]:::computeSecondary
        ler["_load_er()"]:::computeSecondary
        nx["_compute_betweenness<br/>+ _select_and_expand<br/>(NetworkX)"]:::computePrimary
    end

    subgraph sq["SQLite — {book_id}_{model}.db (per-database)"]
        nodes[("nodes<br/>name, type, mentions")]:::dataPrimary
        edges[("edges<br/>src, dst, rel, weight")]:::dataPrimary
        leiden[("leiden_communities<br/>node, community, resolution")]:::dataPrimary
        clusters[("entity_clusters<br/>name → canonical")]:::dataSecondary
        labels[("*_labels<br/>(community / canonical)")]:::dataSecondary
    end

    user -->|tap, drag, modifier| cyc
    user -->|edit controls| rp
    s3 -->|topN, seed, depth| st
    s4 -->|JSON layoutConfig| st
    s5 -->|node/edge style| st
    st --> eff
    eff -->|fetchKG, debounced 250ms| proxy
    s6 -->|cy.png / cy.svg| cyc
    cyc -->|blob URL| pngOut
    cyc -->|svg string| svgOut
    cyc -.uses.-> fcose
    cyc -.uses.-> svg
    kgp -->|elements + stylesheet| cyc
    cyc -->|select / unselect events| kgp

    proxy --> rt
    rt --> lg
    lg --> lb
    lg --> ler
    lb --> nx
    ler --> nx
    lb -->|SELECT| nodes
    lb -->|SELECT| edges
    lb -->|WHERE resolution=?| leiden
    ler -->|name → canonical| clusters
    ler -->|aggregate| nodes
    ler -->|dedup| edges
    ler --> leiden
    lg -.optional.-> labels
    nx -->|JSON KGPayload| rt

    classDef actor              fill:#cbd5e1,stroke:#334155,color:#1e293b,stroke-width:1px
    classDef browserPrimary     fill:#2563eb,stroke:#000000,color:#fff,stroke-width:2px
    classDef browserSecondary   fill:#93c5fd,stroke:#1e3a8a,color:#1e293b,stroke-width:1px
    classDef pluginTier         fill:#dbeafe,stroke:#1e3a8a,color:#1e293b,stroke-width:1px,stroke-dasharray:5 5
    classDef panelTier          fill:#cbd5e1,stroke:#334155,color:#1e293b,stroke-width:1px
    classDef apiPrimary         fill:#7c3aed,stroke:#000000,color:#fff,stroke-width:2px
    classDef computePrimary     fill:#7c3aed,stroke:#000000,color:#fff,stroke-width:2px
    classDef computeSecondary   fill:#c4b5fd,stroke:#4c1d95,color:#1e293b,stroke-width:1px
    classDef infraSecondary     fill:#cbd5e1,stroke:#334155,color:#1e293b,stroke-width:1px
    classDef dataPrimary        fill:#0f766e,stroke:#000000,color:#fff,stroke-width:2px
    classDef dataSecondary      fill:#99f6e4,stroke:#115e59,color:#1e293b,stroke-width:1px
    classDef export             fill:#047857,stroke:#000000,color:#fff,stroke-width:2px

    classDef sgBrowser fill:#dbeafe,stroke:#1e3a8a,color:#1e293b
    classDef sgPanel   fill:#f1f5f9,stroke:#475569,color:#334155
    classDef sgBackend fill:#ede9fe,stroke:#5b21b6,color:#1e293b
    classDef sgData    fill:#ccfbf1,stroke:#115e59,color:#1e293b
    class fe sgBrowser
    class rp sgPanel
    class be sgBackend
    class sq sgData
```

</details>

### How to read the detailed view

- **Solid arrows** are runtime data flow (request, response, render).
- **Dashed `.uses.` arrows** mean "registered as a Cytoscape plugin" —
  no data flows along them at request time.
- **Dotted arrows** mark read-only inspection paths
  (`Selection` reads from `KGPage`; it doesn't drive state changes).
- **Subgraph color** keys the layer: blue = Browser, slate = Right
  Panel, violet = Backend, teal = SQLite.

### Critical paths to remember

- **`base` vs `er`** — only the loader branch (`_load_base` /
  `_load_er`) differs. `base` returns raw NER entities; `er` returns
  cluster heads with edges deduplicated and node metadata aggregated.
  Everything downstream (centrality, BFS expansion, payload assembly)
  is identical.
- **All graph computation is in-memory NetworkX**, not SQL. Betweenness
  centrality is computed on the *full* loaded graph, then BFS-expansion
  + min-degree pruning narrow the result to seeds × depth before the
  payload is sent.
- **Constraints are JSON-resident.** The `Constraints` panel section
  mutates `layoutConfig` (a string) directly; the `parsedConstraints`
  view derives from it each render. This is why a Pin click immediately
  appears in the JSON textarea — the JSON is the source of truth, not a
  shadowed React state.
- **Export bypasses the backend.** Cytoscape's `cy.png()` and
  `cy.svg()` operate on the live in-browser graph; PNG and SVG
  downloads do not call any API endpoint.
