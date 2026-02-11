# Node.js Cookbook

## Setup (better-sqlite3)

```javascript
import Database from "better-sqlite3";
import { load } from "sqlite-muninn";

const db = new Database(":memory:");
load(db);
```

## Setup (node:sqlite — Node 22.5+)

```javascript
import { DatabaseSync } from "node:sqlite";
import { load } from "sqlite-muninn";

const db = new DatabaseSync(":memory:", { allowExtension: true });
load(db);
```

## Vector Encoding Helper

```javascript
function encodeVector(values) {
  return Buffer.from(new Float32Array(values).buffer);
}

function decodeVector(blob, dim) {
  return new Float32Array(blob.buffer, blob.byteOffset, dim);
}
```

## HNSW Search Endpoint (Express)

```javascript
import express from "express";
import Database from "better-sqlite3";
import { load } from "sqlite-muninn";

const app = express();
const db = new Database("vectors.db");
load(db);

app.use(express.json());

const searchStmt = db.prepare(
  "SELECT rowid, distance FROM embeddings WHERE vector MATCH ? AND k = ?"
);

app.post("/search", (req, res) => {
  const { vector, k = 10 } = req.body;
  const blob = Buffer.from(new Float32Array(vector).buffer);
  const results = searchStmt.all(blob, k);
  res.json(results);
});
```

## Batch Insert with Transactions

```javascript
const dim = 128;
const insert = db.prepare(
  "INSERT INTO my_index(rowid, vector) VALUES (?, ?)"
);

const insertMany = db.transaction((items) => {
  for (const { id, vector } of items) {
    insert.run(id, Buffer.from(new Float32Array(vector).buffer));
  }
});

insertMany(items); // Runs all inserts in a single transaction
```

## Graph Traversal

```javascript
// BFS from a node
const bfs = db.prepare(
  "SELECT * FROM graph_bfs('edges', 'source', 'target', ?)"
);
const reachable = bfs.all("node_1");

// Shortest path between two nodes
const path = db.prepare(
  "SELECT * FROM graph_shortest_path('edges', 'source', 'target', ?, ?)"
);
const route = path.all("node_1", "node_42");
```
