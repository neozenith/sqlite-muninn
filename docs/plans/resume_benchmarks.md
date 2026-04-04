# Benchmark Run Resume Notes

**Left off:** 2026-04-04  
**Branch:** `feat/benchmarks-cloud-support-2`  
**State:** ASG scaled to 0, SQS queue + DLQ purged.

---

## Current Completion

```
2737 / 3886 done
Missing: 30 EMBED + 1119 VSS
```

All other categories (CENTRALITY, COMMUNITY, GRAPH, GRAPH_VT, NODE2VEC) are 100% complete.

---

## Blockers to Fix Before Re-Submitting

### 1. OOM in VSS ground-truth computation — HIGH PRIORITY

**File:** `benchmarks/harness/treatments/vss.py` line 85  
**Function:** `_compute_ground_truth()`

```python
# THIS LINE allocates an (M, N, dim) tensor — full 3D distance matrix in RAM
dists = np.sum((doc_vectors[None, :, :] - query_vectors[:, None, :]) ** 2, axis=2)
```

**Why it OOMs:** Broadcasting creates a temporary `(M=100, N, dim)` float32 array before the `sum` collapses axis 2. At N=50000 + NomicEmbed (768d) that's `100 × 50000 × 768 × 4` = **14.3 GiB**. t3.xlarge only has 16 GB total, and the Python process + HNSW index already consume several GB.

**Fix:** Compute distances query-by-query (loop over M queries), or use `scipy.spatial.distance.cdist` which is memory-efficient. Query-by-query is simplest:

```python
def _compute_ground_truth(doc_vectors, query_vectors, k):
    results = []
    for qv in query_vectors:
        diffs = doc_vectors - qv          # (N, dim) — reused, no extra alloc
        dists = np.einsum('nd,nd->n', diffs, diffs)  # (N,) L2² per doc
        top_k = np.argsort(dists)[:k]
        results.append({int(idx) + 1 for idx in top_k})
    return results
```

Or with `cdist` (allocates only `(M, N)` = `100 × 50000 × 4` = 20 MB):

```python
from scipy.spatial.distance import cdist

def _compute_ground_truth(doc_vectors, query_vectors, k):
    dists = cdist(query_vectors, doc_vectors, metric='sqeuclidean')  # (M, N)
    top_k_indices = np.argsort(dists, axis=1)[:, :k]
    return [{int(idx) + 1 for idx in row} for row in top_k_indices]
```

**Affected permutations:** Primarily N=50000 and N=100000 with NomicEmbed (768d) and BGE-Large (1024d). MiniLM (384d) at N=50000 is borderline (`100 × 50000 × 384 × 4` = 7.3 GiB — may also OOM under memory pressure).

**Failing benchmarks confirmed in CW logs:**
- `vss_muninn-hnsw_NomicEmbed_ag-news_n50000_m16_efc100_efs400`
- `vss_vectorlite-hnsw_BGE-Large_ag-news_n50000_m32_efc200_efs400`

---

### 2. EMBED — 30 missing permutations

These didn't fail — they just weren't picked up (workers exhausted by VSS jobs before getting to them). All `embed_lembed+*` and `embed_muninn_embed+*` on `wealth-of-nations` dataset at N=500, 1000, 2000, 5000. Re-submit after fixing VSS.

---

## Infrastructure State

| Resource | State |
|----------|-------|
| ASG `MuninnBench-feat-benchmarks-cloud-support-2-ASG*` | desired=0 |
| SQS main queue | purged |
| SQS DLQ | purged |
| AMI `ami-0b7a1eab5b8bdd47a` | still valid (commit `9c152a8`) |
| S3 results | 2737 JSONL records intact |

---

## Commits Applied During This Session

| Commit | Fix |
|--------|-----|
| `51dba03` | node2vec: fix 14→13 arg count (removed `weight` col arg) |
| `10c0171` | embed: add `max_tokens` per model, truncate long texts |
| `41cac31` | harness: stop uploading SQLite DB to S3; worker: rm results dir after success |
| `4615d9b` | pyproject: add `sqlite-lembed` to benchmark dep group |
| `55fc0ab` | runner: add `diagnose` subcommand (CloudWatch Logs Insights) |
| `c11cd22` | prep/\_\_init\_\_.py: remove `vectors` import to avoid top-level `llama_cpp` load |
| `50a22bf` | worker: add `shutdown -h now` to EXIT trap so crashed instances self-terminate |

---

## Resumption Checklist

1. **Fix OOM in `_compute_ground_truth`** — replace the `(M, N, dim)` broadcast with `cdist` or per-query loop (see §1 above).

2. **Redeploy CDK stack** — the `shutdown -in-trap` fix (`50a22bf`) is in `worker_user_data.sh` but CDK embeds user-data at deploy time. Re-deploy so new instances pick up the fix:
   ```bash
   npx aws-cdk@latest deploy MuninnBench-feat-benchmarks-cloud-support-2 \
     --app "uv run --group cdk benchmarks/infra/cdk/app.py" \
     -c account=389956346255 -c branch=feat/benchmarks-cloud-support-2 \
     -c ami_id=ami-0b7a1eab5b8bdd47a
   ```

3. **Re-submit missing benchmarks:**
   ```bash
   uv run --no-sync benchmarks/infra/runner.py submit
   ```
   Expected: ~1149 messages enqueued (30 EMBED + 1119 VSS).

4. **Scale ASG up** (if CW alarm hasn't fired yet):
   ```bash
   aws autoscaling set-desired-capacity \
     --auto-scaling-group-name "MuninnBench-feat-benchmarks-cloud-support-2-ASG46ED3070-F9uArT8rQ2aI" \
     --desired-capacity 5 \
     --region ap-southeast-2
   ```

5. **Monitor** — `uv run --no-sync benchmarks/infra/runner.py diagnose --minutes 30`

---

## Known Failure Modes (for reference)

| Symptom | Root Cause | Fix |
|---------|-----------|-----|
| Instances InService but not heartbeating for hours | Worker script exited (crash/circuit-breaker) before `shutdown -h now`; EXIT trap deleted heartbeat but didn't shut down | Fixed in `50a22bf` — `shutdown` now in trap |
| OOM: `Unable to allocate N GiB for array (100, 50000, dim)` | `_compute_ground_truth` broadcasts full `(M, N, dim)` tensor | Fix `cdist` approach in §1 above |
| `ModuleNotFoundError: No module named 'llama_cpp'` on `submit` | `prep/__init__.py` imported `vectors.py` which has top-level `llama_cpp` import | Fixed in `c11cd22` |
| `wrong number of arguments to function node2vec_train()` | Python sent 14 args; C function takes 13 | Fixed in `51dba03` |
| `muninn_embed()` fails: token count exceeds context | Long texts exceed model max tokens (MiniLM=512, BGE=512) | Fixed in `10c0171` |
| Disk full on worker | Harness was uploading + keeping SQLite DBs; accumulate indefinitely | Fixed in `41cac31` |
| `ModuleNotFoundError: No module named 'sqlite_lembed'` | Missing from `benchmark` dep group | Fixed in `4615d9b` |
