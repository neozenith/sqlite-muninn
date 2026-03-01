/**
 * muninn WASM Demo — App logic
 *
 * Loads the muninn WASM module, fetches the demo database manifest,
 * lets the user pick a database, then wires up:
 *   search → embedding → HNSW → graph traversal → visualization
 *
 * 3D coordinates come from pre-calculated UMAP tables (*_umap) in the database.
 *
 * Three-panel search:
 *   Left:   FTS5 full-text results (instant, no embedding needed)
 *   Center: HNSW embedding space (3D UMAP point cloud, rank-colored)
 *   Right:  Knowledge graph (CTE: VSS → anchor → BFS → filter)
 *
 * Model-aware embedding:
 *   MiniLM    → Xenova/all-MiniLM-L6-v2, symmetric (no prefix)
 *   NomicEmbed → nomic-ai/nomic-embed-text-v1.5, asymmetric ("search_query: " prefix)
 */

// ── Embedding model configs ────────────────────────────────────────────
// NomicEmbed is asymmetric: documents are indexed with "search_document: "
// prefix. Queries need "search_query: " to land in the same embedding subspace.
const WASM_EMBEDDING_MODELS = {
  MiniLM: {
    onnxModelId: "Xenova/all-MiniLM-L6-v2",
    queryPrefix: "",
    quantized: false, // fp32 for parity with Python sentence-transformers
  },
  NomicEmbed: {
    onnxModelId: "nomic-ai/nomic-embed-text-v1.5",
    queryPrefix: "search_query: ",
    quantized: false,
  },
};

// Map raw manifest model slugs to WASM_EMBEDDING_MODELS keys
const MODEL_SLUG_MAP = {
  MiniLM: "MiniLM",
  "all-MiniLM-L6-v2": "MiniLM",
  NomicEmbed: "NomicEmbed",
  "nomic-ai/nomic-embed-text-v1.5": "NomicEmbed",
  "nomic-embed-text-v1.5.Q8_0.gguf": "NomicEmbed",
};

// Maximum DB size loadable in browser (200 MB). Larger databases (e.g. sessions_demo
// at 1.9 GB) cannot be fetched and written into Emscripten's virtual filesystem.
const MAX_WASM_DB_BYTES = 200 * 1024 * 1024;

// ── Global State ──────────────────────────────────────────────────────

let db = null; // SQLite WASM database pointer
let sqlite = null; // SQLite WASM API wrappers
let Module = null; // Emscripten module
let sentenceEmbedder = null; // Transformers.js pipeline
let deckInstance = null; // Deck.GL instance
let cyInstance = null; // Cytoscape instance
let dbSchema = null; // Discovered schema info
let searchDebounce = null;

let manifestDatabases = []; // All databases from manifest.json
let currentDbEntry = null; // Currently loaded manifest entry
let currentDbFilename = null; // Filename written into Emscripten FS
let activeModelKey = "MiniLM"; // Current WASM_EMBEDDING_MODELS key
let queryPrefix = ""; // Active query prefix (set from model config)

// ── Status Management ─────────────────────────────────────────────────

function setStatus(id, status) {
  const el = document.getElementById(`status-${id}`);
  if (el) el.dataset.status = status;
}

// ── Utility ───────────────────────────────────────────────────────────

function formatBytes(bytes) {
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / 1024 / 1024).toFixed(0)} MB`;
  return `${(bytes / 1024 / 1024 / 1024).toFixed(1)} GB`;
}

async function waitFor(check, name, maxAttempts = 100) {
  let attempts = 0;
  while (!check() && attempts < maxAttempts) {
    await new Promise((r) => setTimeout(r, 200));
    attempts++;
  }
  if (!check()) {
    throw new Error(`${name} failed to load after ${maxAttempts * 200}ms`);
  }
}

// ── SQLite WASM Module Initialization ────────────────────────────────
// Initializes the Emscripten module and wraps the C API.
// Does NOT open a database — call loadDatabase() for that.

async function initWasmModule() {
  setStatus("wasm", "loading");
  try {
    Module = await createMuninnSQLite();

    sqlite = {
      open: Module.cwrap("sqlite3_open", "number", ["string", "number"]),
      close: Module.cwrap("sqlite3_close", "number", ["number"]),
      exec: Module.cwrap("sqlite3_exec", "number", [
        "number",
        "string",
        "number",
        "number",
        "number",
      ]),
      errmsg: Module.cwrap("sqlite3_errmsg", "string", ["number"]),
      prepare_v2: Module.cwrap("sqlite3_prepare_v2", "number", [
        "number",
        "string",
        "number",
        "number",
        "number",
      ]),
      step: Module.cwrap("sqlite3_step", "number", ["number"]),
      finalize: Module.cwrap("sqlite3_finalize", "number", ["number"]),
      column_text: Module.cwrap("sqlite3_column_text", "string", [
        "number",
        "number",
      ]),
      column_int: Module.cwrap("sqlite3_column_int", "number", [
        "number",
        "number",
      ]),
      column_double: Module.cwrap("sqlite3_column_double", "number", [
        "number",
        "number",
      ]),
      column_blob: Module.cwrap("sqlite3_column_blob", "number", [
        "number",
        "number",
      ]),
      column_bytes: Module.cwrap("sqlite3_column_bytes", "number", [
        "number",
        "number",
      ]),
      column_count: Module.cwrap("sqlite3_column_count", "number", ["number"]),
      column_name: Module.cwrap("sqlite3_column_name", "string", [
        "number",
        "number",
      ]),
      bind_blob: Module.cwrap("sqlite3_bind_blob", "number", [
        "number",
        "number",
        "number",
        "number",
        "number",
      ]),
      bind_text: Module.cwrap("sqlite3_bind_text", "number", [
        "number",
        "number",
        "string",
        "number",
        "number",
      ]),
      bind_int: Module.cwrap("sqlite3_bind_int", "number", [
        "number",
        "number",
        "number",
      ]),
      reset: Module.cwrap("sqlite3_reset", "number", ["number"]),
      free: Module.cwrap("sqlite3_free", "void", ["number"]),
    };

    // Register muninn as an auto-extension (must be called before sqlite3_open)
    Module.ccall("sqlite3_wasm_extra_init", "number", ["string"], [null]);

    setStatus("wasm", "ready");
    console.log("WASM module initialized");
    return true;
  } catch (err) {
    setStatus("wasm", "error");
    console.error("WASM init failed:", err);
    throw err;
  }
}

// ── Database Loading ───────────────────────────────────────────────────
// Fetches a demo database from /demos/{file} and opens it in WASM SQLite.
// Called on initial load and whenever the user switches databases.

async function loadDatabase(manifestEntry) {
  const hint = document.getElementById("search-hint");
  const sizeBytes = manifestEntry.size_bytes || 0;

  if (sizeBytes > MAX_WASM_DB_BYTES) {
    hint.textContent = `${manifestEntry.label} is ${formatBytes(sizeBytes)} — too large for browser loading (max ${formatBytes(MAX_WASM_DB_BYTES)}). FTS and graph features require a smaller database.`;
    hint.classList.add("text-amber-400");
    document.getElementById("search-input").disabled = true;
    return false;
  }

  hint.classList.remove("text-amber-400", "text-red-400");
  hint.textContent = `Loading ${manifestEntry.label} (${formatBytes(sizeBytes)})…`;

  try {
    // Close previous DB if open
    if (db !== null) {
      sqlite.close(db);
      db = null;
    }
    if (currentDbFilename) {
      try {
        Module.FS.unlink("/" + currentDbFilename);
      } catch (_) {
        // File may not exist — ignore
      }
      currentDbFilename = null;
    }

    const filename = manifestEntry.file;
    const url = "/demos/" + filename;

    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`HTTP ${response.status} fetching ${url}`);
    }
    const arrayBuffer = await response.arrayBuffer();
    const data = new Uint8Array(arrayBuffer);

    Module.FS.writeFile("/" + filename, data);
    currentDbFilename = filename;

    const dbPtrPtr = Module._malloc(4);
    const rc = sqlite.open("/" + filename, dbPtrPtr);
    if (rc !== 0) {
      Module._free(dbPtrPtr);
      throw new Error(`sqlite3_open failed: rc=${rc}`);
    }
    db = Module.getValue(dbPtrPtr, "i32");
    Module._free(dbPtrPtr);

    currentDbEntry = manifestEntry;
    console.log(`Database loaded: ${manifestEntry.label}`);
    return true;
  } catch (err) {
    hint.textContent = `Failed to load database: ${err.message}`;
    hint.classList.add("text-red-400");
    console.error("Database load failed:", err);
    return false;
  }
}

// ── SQL Query Helpers ─────────────────────────────────────────────────

const SQLITE_ROW = 100;

function query(sql) {
  const stmtPtrPtr = Module._malloc(4);
  const rc = sqlite.prepare_v2(db, sql, -1, stmtPtrPtr, 0);
  if (rc !== 0) {
    Module._free(stmtPtrPtr);
    throw new Error(`prepare failed (${rc}): ${sqlite.errmsg(db)}`);
  }
  const stmt = Module.getValue(stmtPtrPtr, "i32");
  Module._free(stmtPtrPtr);

  const cols = sqlite.column_count(stmt);
  const colNames = [];
  for (let i = 0; i < cols; i++) {
    colNames.push(sqlite.column_name(stmt, i));
  }

  const rows = [];
  while (sqlite.step(stmt) === SQLITE_ROW) {
    const row = {};
    for (let i = 0; i < cols; i++) {
      row[colNames[i]] = sqlite.column_text(stmt, i);
    }
    rows.push(row);
  }

  sqlite.finalize(stmt);
  return { columns: colNames, rows };
}

/**
 * Execute a parameterized query with bindings.
 * bindings: array of { index, type, value } where type is 'text', 'int', or 'blob'.
 * For blob: value = { ptr, size } (caller manages memory).
 */
function queryBound(sql, bindings) {
  const stmtPtrPtr = Module._malloc(4);
  const rc = sqlite.prepare_v2(db, sql, -1, stmtPtrPtr, 0);
  if (rc !== 0) {
    Module._free(stmtPtrPtr);
    throw new Error(`prepare failed (${rc}): ${sqlite.errmsg(db)}`);
  }
  const stmt = Module.getValue(stmtPtrPtr, "i32");
  Module._free(stmtPtrPtr);

  for (const b of bindings) {
    let brc;
    if (b.type === "text") {
      brc = sqlite.bind_text(stmt, b.index, b.value, -1, 0);
    } else if (b.type === "int") {
      brc = sqlite.bind_int(stmt, b.index, b.value);
    } else if (b.type === "blob") {
      // SQLITE_TRANSIENT = -1
      brc = sqlite.bind_blob(stmt, b.index, b.value.ptr, b.value.size, -1);
    }
    if (brc !== 0) {
      sqlite.finalize(stmt);
      throw new Error(`bind (idx=${b.index}): ${sqlite.errmsg(db)}`);
    }
  }

  const cols = sqlite.column_count(stmt);
  const colNames = [];
  for (let i = 0; i < cols; i++) {
    colNames.push(sqlite.column_name(stmt, i));
  }

  const rows = [];
  while (sqlite.step(stmt) === SQLITE_ROW) {
    const row = {};
    for (let i = 0; i < cols; i++) {
      row[colNames[i]] = sqlite.column_text(stmt, i);
    }
    rows.push(row);
  }

  sqlite.finalize(stmt);
  return { columns: colNames, rows };
}

// ── Schema Discovery ──────────────────────────────────────────────────

function discoverSchema() {
  const tables = query(
    "SELECT name, type FROM sqlite_master WHERE type IN ('table','view') ORDER BY name",
  );
  const hnswConfigs = [];

  for (const t of tables.rows) {
    if (t.name.endsWith("_config")) {
      const base = t.name.replace(/_config$/, "");
      try {
        const cfg = query(`SELECT key, value FROM "${t.name}"`);
        const config = {};
        for (const r of cfg.rows) config[r.key] = r.value;
        config._base = base;
        hnswConfigs.push(config);
      } catch (_) {
        // Not an HNSW config table
      }
    }
  }

  const counts = {};
  for (const name of ["chunks", "entities", "relations", "edges", "nodes"]) {
    try {
      const r = query(`SELECT COUNT(*) as n FROM "${name}"`);
      counts[name] = parseInt(r.rows[0].n);
    } catch (_) {
      counts[name] = 0;
    }
  }

  const umapTables = tables.rows
    .filter((t) => t.name.endsWith("_umap"))
    .map((t) => t.name);

  dbSchema = { tables: tables.rows, hnswConfigs, counts, umapTables };

  const statusEl = document.getElementById("db-status");
  statusEl.textContent =
    `${currentDbEntry ? currentDbEntry.label : "Database"}: ` +
    `${counts.chunks} chunks, ${counts.entities} entities, ` +
    `${counts.relations} relations, ${hnswConfigs.length} HNSW indexes, ` +
    `${umapTables.length} UMAP tables`;

  console.log("Schema discovered:", dbSchema);
  return dbSchema;
}

// ── Manifest & DB Selector ────────────────────────────────────────────

async function initManifest() {
  try {
    const response = await fetch("/demos/manifest.json");
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();
    manifestDatabases = data.databases || [];
    console.log(`Manifest loaded: ${manifestDatabases.length} databases`);
    return manifestDatabases;
  } catch (err) {
    console.error("Failed to load manifest:", err);
    manifestDatabases = [];
    return [];
  }
}

function renderDbSelector(databases) {
  const container = document.getElementById("db-selector-container");
  if (!databases.length) {
    container.innerHTML =
      '<span class="text-xs text-red-400">No databases found in /demos/manifest.json</span>';
    return;
  }

  const select = document.createElement("select");
  select.id = "wasm-db-selector";
  select.className =
    "h-7 rounded-md border border-gray-700 bg-gray-900 px-2 text-xs text-gray-100 focus:outline-none focus:border-indigo-500 disabled:opacity-50";
  select.setAttribute("aria-label", "Select database");

  for (const db of databases) {
    const option = document.createElement("option");
    option.value = db.id;
    option.textContent = db.label;
    // Warn about oversized databases
    if ((db.size_bytes || 0) > MAX_WASM_DB_BYTES) {
      option.textContent += " ⚠ too large";
    }
    select.appendChild(option);
  }

  // Restore last selection from localStorage
  const stored = localStorage.getItem("muninn-selected-db");
  if (stored && databases.some((d) => d.id === stored)) {
    select.value = stored;
  }

  select.addEventListener("change", (e) => {
    const entry = databases.find((d) => d.id === e.target.value);
    if (entry) onDbChange(entry, select);
  });

  container.innerHTML = "";
  container.appendChild(select);
}

async function onDbChange(manifestEntry, selectEl) {
  if (selectEl) selectEl.disabled = true;
  disableSearch("Loading database…");
  clearResults();

  const modelKey = MODEL_SLUG_MAP[manifestEntry.model] || "MiniLM";
  const ok = await loadDatabase(manifestEntry);

  if (ok) {
    discoverSchema();

    // Reload embedding model only if it changed
    if (modelKey !== activeModelKey || sentenceEmbedder === null) {
      sentenceEmbedder = null;
      setStatus("transformers", "loading");
      await initTransformers(modelKey);
    }

    localStorage.setItem("muninn-selected-db", manifestEntry.id);
    enableSearch();
  }

  if (selectEl) selectEl.disabled = false;
}

// ── Transformers.js Initialization ────────────────────────────────────

async function initTransformers(modelKey) {
  activeModelKey = modelKey || "MiniLM";
  const cfg = WASM_EMBEDDING_MODELS[activeModelKey] || WASM_EMBEDDING_MODELS.MiniLM;
  queryPrefix = cfg.queryPrefix;

  setStatus("transformers", "loading");
  try {
    await waitFor(() => window.transformers, "Transformers.js");

    sentenceEmbedder = await window.transformers.pipeline(
      "feature-extraction",
      cfg.onnxModelId,
      { pooling: "mean", normalize: true, quantized: cfg.quantized },
    );

    setStatus("transformers", "ready");
    console.log(
      `Embedding model loaded: ${cfg.onnxModelId} (prefix: "${cfg.queryPrefix || "none"}")`,
    );
    return sentenceEmbedder;
  } catch (err) {
    setStatus("transformers", "error");
    console.error("Transformers.js init failed:", err);
    // Non-fatal: FTS search still works without embeddings
    sentenceEmbedder = null;
    return null;
  }
}

async function generateEmbedding(text) {
  if (!sentenceEmbedder) return null;
  // Prepend query prefix for asymmetric models (NomicEmbed: "search_query: ").
  // The indexed documents were stored with "search_document: " prefix, so
  // using the correct query prefix is required for meaningful similarity scores.
  const prefixedText = queryPrefix ? queryPrefix + text : text;
  const output = await sentenceEmbedder(prefixedText, {
    pooling: "mean",
    normalize: true,
  });
  return Array.from(output.data);
}

// ── Search Enable/Disable ─────────────────────────────────────────────

function enableSearch() {
  const input = document.getElementById("search-input");
  const hint = document.getElementById("search-hint");
  input.disabled = false;
  hint.classList.remove("text-amber-400", "text-red-400");
  const modelCfg =
    WASM_EMBEDDING_MODELS[activeModelKey] || WASM_EMBEDDING_MODELS.MiniLM;
  const label = currentDbEntry ? currentDbEntry.label : "database";
  hint.textContent = sentenceEmbedder
    ? `Searching "${label}" — FTS + ${modelCfg.onnxModelId.split("/").pop()} embeddings + graph.`
    : `Searching "${label}" — FTS only (embedding model unavailable).`;
}

function disableSearch(message) {
  const input = document.getElementById("search-input");
  const hint = document.getElementById("search-hint");
  input.disabled = true;
  hint.textContent = message || "Loading…";
}

function clearResults() {
  document.getElementById("results-count").textContent = "0 results";
  document.getElementById("deckgl-count").textContent = "0 points";
  document.getElementById("cytoscape-count").textContent = "0 nodes";
  document.getElementById("results-list")
    .querySelectorAll(".result-card")
    .forEach((c) => c.remove());
  const rp = document.getElementById("results-placeholder");
  if (rp) rp.style.display = "";
  const er = document.getElementById("embedding-results");
  if (er) {
    er.innerHTML = "";
    er.classList.add("hidden");
  }
  const dp = document.getElementById("deckgl-placeholder");
  if (dp) dp.style.display = "";
  if (cyInstance) cyInstance.elements().remove();
  const gc = document.getElementById("graph-controls");
  if (gc) gc.classList.add("hidden");
}

// ── Rank-Based Colors ─────────────────────────────────────────────────

function rankColor(rank, total) {
  if (rank === 0) return [255, 60, 60];
  if (total <= 1) return [255, 60, 60];
  const t = (rank - 1) / (total - 1);
  return [
    Math.round(255 + (140 - 255) * t),
    Math.round(140 + (60 - 140) * t),
    Math.round(50 + (220 - 50) * t),
  ];
}

// ── Deck.GL Initialization ────────────────────────────────────────────

function initDeckGL() {
  setStatus("deckgl", "loading");
  try {
    const container = document.getElementById("deckgl-container");
    if (!container || !window.deck) {
      throw new Error("Deck.GL container or library not found");
    }

    const canvas = document.createElement("canvas");
    canvas.style.width = "100%";
    canvas.style.height = "100%";
    container.appendChild(canvas);

    const lightingEffect = new deck.LightingEffect({
      ambientLight: new deck.AmbientLight({
        color: [255, 255, 255],
        intensity: 0.3,
      }),
      directionalLight: new deck.DirectionalLight({
        color: [255, 255, 255],
        intensity: 0.8,
        direction: [-1, -1, -2],
      }),
    });

    deckInstance = new deck.Deck({
      canvas: canvas,
      width: container.clientWidth,
      height: container.clientHeight || 400,
      views: [new deck.OrbitView({ orbitAxis: "Y", fov: 50 })],
      initialViewState: { target: [0, 0, 0], zoom: 1 },
      controller: true,
      effects: [lightingEffect],
      layers: [],
      getTooltip: ({ object }) => {
        if (!object) return null;
        const pct = (object.similarity * 100).toFixed(1);
        const text =
          object.chunkText.length > 80
            ? object.chunkText.substring(0, 80) + "..."
            : object.chunkText;
        return {
          html: `<div style="max-width:300px"><div style="font-weight:600; margin-bottom:4px">Chunk #${object.chunkId} &middot; ${pct}%</div><div style="font-size:11px; line-height:1.4">${text || "No text"}</div></div>`,
          style: {
            backgroundColor: "rgba(0,0,0,0.85)",
            color: "white",
            padding: "8px 12px",
            borderRadius: "6px",
            fontSize: "12px",
          },
        };
      },
    });

    window.deckInstance = deckInstance;
    setStatus("deckgl", "ready");
    console.log("Deck.GL initialized");
    return deckInstance;
  } catch (err) {
    setStatus("deckgl", "error");
    console.error("Deck.GL init failed:", err);
    // Non-fatal: WebGL may not work in headless
  }
}

function updateDeckGL(points) {
  if (!deckInstance) return;

  const sphere = new luma.SphereGeometry({ radius: 1, nlat: 10, nlong: 20 });

  let cx = 0,
    cy = 0,
    cz = 0;
  for (const p of points) {
    cx += p.position[0];
    cy += p.position[1];
    cz += p.position[2];
  }
  const n = points.length || 1;
  cx /= n;
  cy /= n;
  cz /= n;

  let maxR = 0;
  for (const p of points) {
    const dx = p.position[0] - cx;
    const dy = p.position[1] - cy;
    const dz = p.position[2] - cz;
    const r = Math.sqrt(dx * dx + dy * dy + dz * dz) + p.radius;
    if (r > maxR) maxR = r;
  }

  const container = document.getElementById("deckgl-container");
  const viewSize =
    Math.min(container.clientWidth, container.clientHeight) || 400;
  const padding = 2.0;
  const effectiveRadius = maxR * padding || 1;
  const zoom = Math.log2(viewSize / (2 * effectiveRadius));

  deckInstance.setProps({
    initialViewState: {
      target: [cx, cy, cz],
      zoom: zoom,
      rotationX: 30,
      rotationOrbit: -30,
    },
    layers: [
      new deck.SimpleMeshLayer({
        id: "embedding-points",
        data: points,
        mesh: sphere,
        getPosition: (d) => d.position,
        getColor: (d) => d.color,
        getTransformMatrix: (d) => [
          d.radius, 0, 0, 0,
          0, d.radius, 0, 0,
          0, 0, d.radius, 0,
          0, 0, 0, 1,
        ],
        pickable: true,
        autoHighlight: true,
        updateTriggers: {
          getPosition: points,
          getColor: points,
          getTransformMatrix: points,
        },
      }),
    ],
  });

  const countEl = document.getElementById("deckgl-count");
  countEl.textContent = `${points.length} points`;

  const placeholder = document.getElementById("deckgl-placeholder");
  if (placeholder) placeholder.style.display = "none";
}

// ── Cytoscape Initialization ──────────────────────────────────────────

function initCytoscape() {
  setStatus("cytoscape", "loading");
  try {
    if (!window.cytoscape) {
      throw new Error("Cytoscape library not found");
    }

    cyInstance = cytoscape({
      container: document.getElementById("cy"),
      elements: [],
      style: [
        {
          selector: "node",
          style: {
            width: "data(size)",
            height: "data(size)",
            label: "data(label)",
            "text-valign": "center",
            "text-halign": "center",
            "font-size": "10px",
            color: "#e5e7eb",
            "text-outline-width": 2,
            "text-outline-color": "#1f2937",
            "background-color": "data(color)",
          },
        },
        {
          selector: "edge",
          style: {
            width: 2,
            "line-color": "#4b5563",
            "target-arrow-color": "#4b5563",
            "target-arrow-shape": "triangle",
            "curve-style": "bezier",
            label: "data(label)",
            "font-size": "8px",
            color: "#6b7280",
            "text-rotation": "autorotate",
          },
        },
        {
          selector: ".highlighted",
          style: {
            "background-color": "#818cf8",
            "line-color": "#818cf8",
            "target-arrow-color": "#818cf8",
          },
        },
        {
          selector: ".query-node",
          style: {
            "background-color": "#f87171",
            "border-width": 3,
            "border-color": "#fca5a5",
          },
        },
      ],
      layout: { name: "grid", rows: 1 },
      zoomingEnabled: true,
      panningEnabled: true,
    });

    window.cyInstance = cyInstance;
    setStatus("cytoscape", "ready");
    console.log("Cytoscape initialized");
    return cyInstance;
  } catch (err) {
    setStatus("cytoscape", "error");
    console.error("Cytoscape init failed:", err);
  }
}

function updateCytoscape(nodes, edges) {
  if (!cyInstance) return;

  cyInstance.elements().remove();

  const elements = [];

  for (const n of nodes) {
    elements.push({
      group: "nodes",
      data: {
        id: n.name,
        label: n.name,
        size: n.isAnchor ? 40 : 20 + Math.min(n.similarity || 0, 1) * 20,
        color: n.isAnchor ? "#f87171" : nodeColor(n.similarity || 0),
      },
      classes: n.isAnchor ? "query-node" : "",
    });
  }

  for (const e of edges) {
    if (
      elements.some((el) => el.data.id === e.src) &&
      elements.some((el) => el.data.id === e.dst)
    ) {
      elements.push({
        group: "edges",
        data: {
          id: `${e.src}-${e.rel}-${e.dst}`,
          source: e.src,
          target: e.dst,
          label: e.rel || "",
        },
      });
    }
  }

  cyInstance.add(elements);
  cyInstance.layout({ name: "cose", animate: false, padding: 30 }).run();
  cyInstance.fit(undefined, 20);

  const countEl = document.getElementById("cytoscape-count");
  countEl.textContent = `${nodes.length} nodes`;

  const controls = document.getElementById("graph-controls");
  if (controls) {
    controls.classList.toggle("hidden", nodes.length === 0);
  }
}

function nodeColor(similarity) {
  if (similarity > 0.5) return "#f59e0b";
  if (similarity > 0.2) return "#8b5cf6";
  return "#6b7280";
}

// ── Graph Layout Controls ─────────────────────────────────────────────

function rerunLayout() {
  if (!cyInstance) return;
  const repulsion = parseInt(document.getElementById("repulsion").value);
  const edgeLength = parseInt(document.getElementById("edge-length").value);
  const gravity = parseFloat(document.getElementById("gravity").value);
  cyInstance
    .layout({
      name: "cose",
      animate: false,
      padding: 30,
      nodeRepulsion: () => repulsion,
      idealEdgeLength: () => edgeLength,
      gravity: gravity,
    })
    .run();
  cyInstance.fit(undefined, 20);
}

function initGraphControls() {
  const sliders = [
    { id: "repulsion", valId: "repulsion-val", fmt: (v) => v },
    { id: "edge-length", valId: "edge-length-val", fmt: (v) => v },
    {
      id: "gravity",
      valId: "gravity-val",
      fmt: (v) => parseFloat(v).toFixed(2),
    },
  ];

  for (const s of sliders) {
    const el = document.getElementById(s.id);
    if (el) {
      el.addEventListener("input", () => {
        document.getElementById(s.valId).textContent = s.fmt(el.value);
      });
    }
  }

  const btn = document.getElementById("run-layout-btn");
  if (btn) btn.addEventListener("click", rerunLayout);
}

// ── FTS5 Search ───────────────────────────────────────────────────────

function performFtsSearch(queryText) {
  try {
    const sanitized = queryText.replace(/[^\w\s]/g, " ");
    const ftsQuery = sanitized
      .trim()
      .split(/\s+/)
      .filter((w) => w.length > 0)
      .map((w) => `"${w}"`)
      .join(" ");
    if (!ftsQuery) return [];

    const results = queryBound(
      `SELECT chunk_id, text FROM chunks WHERE chunk_id IN (
        SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ?1 LIMIT 20
      )`,
      [{ index: 1, type: "text", value: ftsQuery }],
    );

    return results.rows.map((r) => ({
      id: parseInt(r.chunk_id),
      text: r.text,
    }));
  } catch (err) {
    console.warn("FTS search failed:", err.message);
    return [];
  }
}

// ── CTE Graph Query ───────────────────────────────────────────────────

function performGraphSearch(blobPtr, blobSize, broadK, bfsDepth) {
  const NODE_SQL = `
    WITH
    vss_matches AS (
        SELECT rowid AS vec_rowid, distance AS cosine_distance
        FROM entities_vec
        WHERE vector MATCH ?1 AND k = ?2
    ),
    vss_entities AS (
        SELECT m.name, v.cosine_distance, (1.0 - v.cosine_distance) AS similarity
        FROM vss_matches v
        JOIN entity_vec_map m ON m.rowid = v.vec_rowid
    ),
    anchor AS (
        SELECT name FROM vss_entities ORDER BY cosine_distance ASC LIMIT 1
    ),
    bfs_neighbors AS (
        SELECT node, depth
        FROM graph_bfs
        WHERE edge_table = 'relations' AND src_col = 'src' AND dst_col = 'dst'
          AND start_node = (SELECT name FROM anchor)
          AND max_depth = ?3 AND direction = 'both'
    ),
    scored AS (
        SELECT b.node, b.depth,
               COALESCE(v.cosine_distance, 1.0) AS cosine_distance,
               COALESCE(v.similarity, 0.0) AS similarity
        FROM bfs_neighbors b
        LEFT JOIN vss_entities v ON v.name = b.node
    )
    SELECT node AS name, depth, similarity FROM scored
  `;

  try {
    const nodeResult = queryBound(NODE_SQL, [
      { index: 1, type: "blob", value: { ptr: blobPtr, size: blobSize } },
      { index: 2, type: "text", value: String(broadK) },
      { index: 3, type: "text", value: String(bfsDepth) },
    ]);

    const nodes = nodeResult.rows.map((row) => ({
      name: row.name,
      depth: parseInt(row.depth),
      similarity: parseFloat(row.similarity),
      isAnchor: parseInt(row.depth) === 0,
    }));

    let edges = [];
    if (nodes.length > 1) {
      const names = nodes
        .map((n) => `'${n.name.replace(/'/g, "''")}'`)
        .join(",");
      const edgeResult = query(
        `SELECT src, rel_type, dst FROM relations
         WHERE src IN (${names}) AND dst IN (${names})`,
      );
      edges = edgeResult.rows.map((row) => ({
        src: row.src,
        dst: row.dst,
        rel: row.rel_type,
      }));
    }

    console.log(`Graph search: ${nodes.length} nodes, ${edges.length} edges`);
    return { nodes, edges };
  } catch (err) {
    console.warn("Graph search failed:", err.message);
    return { nodes: [], edges: [] };
  }
}

// ── Search Flow ───────────────────────────────────────────────────────

async function performSearch(queryText) {
  if (!queryText.trim()) return;

  const spinner = document.getElementById("search-spinner");
  spinner.classList.remove("hidden");

  try {
    // Path 1: FTS5 search (instant — no embedding needed)
    const ftsResults = performFtsSearch(queryText);
    showFtsResults(ftsResults);

    // Path 2 & 3: HNSW + graph (require embedding)
    if (!sentenceEmbedder) {
      console.log("No embedding model — FTS only");
      return;
    }

    const queryEmbedding = await generateEmbedding(queryText);
    if (!queryEmbedding) return;

    console.log(
      `Generated ${queryEmbedding.length}-dim embedding (model: ${activeModelKey}, prefix: "${queryPrefix || "none"}")`,
    );

    const floatArray = new Float32Array(queryEmbedding);
    const blobPtr = Module._malloc(floatArray.byteLength);
    Module.HEAPF32.set(floatArray, blobPtr >> 2);

    try {
      // HNSW vector search on chunks_vec
      const searchSql = `
        SELECT rowid, distance
        FROM chunks_vec
        WHERE vector MATCH ?
        AND k = 20
      `;
      const stmtPtrPtr = Module._malloc(4);
      let rc = sqlite.prepare_v2(db, searchSql, -1, stmtPtrPtr, 0);
      if (rc !== 0) {
        Module._free(stmtPtrPtr);
        throw new Error(`search prepare: ${sqlite.errmsg(db)}`);
      }
      const stmt = Module.getValue(stmtPtrPtr, "i32");
      Module._free(stmtPtrPtr);

      rc = sqlite.bind_blob(stmt, 1, blobPtr, floatArray.byteLength, -1);
      if (rc !== 0) {
        sqlite.finalize(stmt);
        throw new Error(`bind_blob: ${sqlite.errmsg(db)}`);
      }

      const searchResults = [];
      while (sqlite.step(stmt) === SQLITE_ROW) {
        searchResults.push({
          rowid: parseInt(sqlite.column_text(stmt, 0)),
          distance: parseFloat(sqlite.column_text(stmt, 1)),
        });
      }
      sqlite.finalize(stmt);

      console.log(
        `HNSW search: ${searchResults.length} results` +
          (searchResults.length > 0
            ? `, top similarity: ${(1 - searchResults[0].distance).toFixed(3)}`
            : ""),
      );

      if (searchResults.length > 0) {
        const rowids = searchResults.map((r) => r.rowid).join(",");
        const chunks = query(
          `SELECT chunk_id, text FROM chunks WHERE chunk_id IN (${rowids})`,
        );
        const chunkMap = {};
        for (const c of chunks.rows) chunkMap[c.chunk_id] = c.text;

        // Read 3D UMAP coords from pre-computed table
        const umapMap = {};
        try {
          const umapResults = query(
            `SELECT id, x3d, y3d, z3d FROM chunks_vec_umap WHERE id IN (${rowids})`,
          );
          for (const u of umapResults.rows) {
            umapMap[u.id] = [
              parseFloat(u.x3d),
              parseFloat(u.y3d),
              parseFloat(u.z3d),
            ];
          }
        } catch (_) {
          // No UMAP table — use fallback positions
        }

        showEmbeddingResults(searchResults, chunkMap);

        const total = searchResults.length;
        const points = searchResults.map((sr, rank) => {
          const coords = umapMap[sr.rowid];
          const similarity = 1 - sr.distance;
          return {
            position: coords || [
              Math.cos((rank / total) * Math.PI * 2) * 30,
              (similarity - 0.5) * 60,
              Math.sin((rank / total) * Math.PI * 2) * 30,
            ],
            color: rankColor(rank, total),
            radius: 3 + similarity * 8,
            chunkId: sr.rowid,
            chunkText: chunkMap[sr.rowid] || "",
            similarity: similarity,
          };
        });
        updateDeckGL(points);
      }

      // CTE Graph search on entities_vec
      const graph = performGraphSearch(
        blobPtr,
        floatArray.byteLength,
        50,
        1,
      );
      updateCytoscape(graph.nodes, graph.edges);
    } finally {
      Module._free(blobPtr);
    }
  } catch (err) {
    console.error("Search failed:", err);
  } finally {
    spinner.classList.add("hidden");
  }
}

// ── Results Panels ────────────────────────────────────────────────────

function showFtsResults(results) {
  const list = document.getElementById("results-list");
  const countEl = document.getElementById("results-count");
  const placeholder = document.getElementById("results-placeholder");

  if (results.length === 0) {
    countEl.textContent = "0 results";
    if (placeholder) placeholder.style.display = "";
    list.querySelectorAll(".result-card").forEach((c) => c.remove());
    return;
  }

  if (placeholder) placeholder.style.display = "none";
  countEl.textContent = `${results.length} results`;

  list.querySelectorAll(".result-card").forEach((c) => c.remove());

  for (const r of results) {
    const card = document.createElement("div");
    card.className = "result-card";
    card.innerHTML = `
      <div class="flex items-center justify-between mb-1">
        <span class="text-xs text-indigo-400 font-medium">Chunk #${r.id}</span>
      </div>
      <p class="text-sm text-gray-300">${escapeHtml(r.text)}</p>
    `;
    list.appendChild(card);
  }
}

function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

function showEmbeddingResults(searchResults, chunkMap) {
  const container = document.getElementById("embedding-results");
  if (!container) return;

  container.innerHTML = "";

  if (searchResults.length === 0) {
    container.classList.add("hidden");
    return;
  }

  container.classList.remove("hidden");

  for (const sr of searchResults) {
    const similarity = 1 - sr.distance;
    const text = chunkMap[sr.rowid] || "";
    const card = document.createElement("div");
    card.className = "result-card";
    card.innerHTML = `
      <div class="flex items-center justify-between mb-1">
        <span class="text-xs text-indigo-400 font-medium">Chunk #${sr.rowid}</span>
        <span class="text-xs font-mono ${similarity > 0.5 ? "text-amber-400" : similarity > 0.2 ? "text-purple-400" : "text-gray-500"}">${(similarity * 100).toFixed(1)}%</span>
      </div>
      <p class="text-xs text-gray-400">${escapeHtml(text)}</p>
    `;
    container.appendChild(card);
  }
}

// ── Initialization ────────────────────────────────────────────────────

async function initialize() {
  console.log("Initializing muninn WASM demo…");

  try {
    // Phase 1: WASM module (critical path — must succeed before anything else)
    await initWasmModule();

    // Phase 2: Visualization libs (non-blocking — WebGL failures are OK)
    initDeckGL();
    initCytoscape();
    initGraphControls();

    // Phase 3: Fetch manifest and render DB selector
    const databases = await initManifest();
    renderDbSelector(databases);

    if (!databases.length) {
      disableSearch("No databases found. Run the demo builder first.");
      return;
    }

    // Phase 4: Load the first (or previously selected) database
    const stored = localStorage.getItem("muninn-selected-db");
    const initialEntry =
      databases.find((d) => d.id === stored) || databases[0];

    // Update selector to reflect choice
    const sel = document.getElementById("wasm-db-selector");
    if (sel) sel.value = initialEntry.id;

    const modelKey = MODEL_SLUG_MAP[initialEntry.model] || "MiniLM";
    const ok = await loadDatabase(initialEntry);
    if (!ok) return; // loadDatabase shows its own error

    discoverSchema();

    // Phase 5: Load embedding model (slow — first load downloads ~90-300 MB)
    await initTransformers(modelKey);

    // Phase 6: Enable search
    localStorage.setItem("muninn-selected-db", initialEntry.id);
    enableSearch();

    const input = document.getElementById("search-input");
    input.addEventListener("input", (e) => {
      clearTimeout(searchDebounce);
      searchDebounce = setTimeout(() => {
        performSearch(e.target.value);
      }, 300);
    });

    console.log("muninn WASM demo ready");
  } catch (err) {
    console.error("Initialization failed:", err);
    const hint = document.getElementById("search-hint");
    hint.textContent = `Initialization failed: ${err.message}`;
    hint.classList.add("text-red-400");
  }
}

document.addEventListener("DOMContentLoaded", initialize);
