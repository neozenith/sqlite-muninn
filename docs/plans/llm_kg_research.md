# LLM-Based KG Extraction — Research Snapshot & Model Catalog

> Research conducted: 2026-03-02
> Covers: SOTA NER / RE / ER papers, open-source KG frameworks, GGUF chat model
> catalog, and extensions needed for `benchmarks.harness prep gguf`.

---

## 1. State of the Art — NER (2025–2026)

### ReasoningNER — AAAI 2026

**Paper:** [arXiv 2511.11978](https://arxiv.org/abs/2511.11978)
**Code:** `HuiResearch/ReasoningIE`

Shifts NER from implicit pattern matching to explicit verifiable reasoning. Three
stages:
1. CoT dataset generation (task-relevant reasoning chains)
2. CoT fine-tuning (model generates rationale before entity prediction)
3. RL reward-signal enhancement (comprehensive reward over reasoning steps)

**Result:** Beats GPT-4 by **12.3 F1 points** zero-shot on the OOD NER benchmark.
Key insight: the explicit reasoning chain is what makes zero-shot generalisation
robust across novel entity types.

### GLiNER2 — Unified Four-Task Extraction

**Package:** `gliner2` (pip) — `fastino-ai/GLiNER2` on GitHub.
**Note:** This is a completely separate project from `urchade/GLiNER`.

A single 205M–340M parameter model unifying **four tasks** in one architecture:

| Task | API |
|---|---|
| Named Entity Recognition | `extractor.extract_entities(text, labels)` |
| **Relation Extraction** | `extractor.extract_relations(text, relation_types)` |
| Text Classification | `extractor.classify_text(text, schema)` |
| Structured Data Extraction | `extractor.extract_json(text, schema)` |

This directly replaces both the GLiNER NER phase AND the GLiREL RE phase with a
single model load. See `docs/plans/gliner2_re_upgrade.md` for the full migration
plan for `demo_builder` and `sessions_demo`.

### Assessment of Generative NER (2026)

**Paper:** [arXiv 2601.17898](https://arxiv.org/html/2601.17898)

Comprehensive comparative survey. Key finding: encoder models (GLiNER family) still
dominate on **speed and resource** efficiency; LLMs win on **zero-shot
generalisation to novel entity types** and low-resource domains. The practical
sweet-spot is a hybrid: encoder model for bulk extraction, LLM for disambiguation
on low-confidence spans.

### Additional Papers

| Paper | Link | Takeaway |
|---|---|---|
| ZERONER — zero-shot NER with KB retrieval | [ACL 2025](https://aclanthology.org/2025.findings-acl.805.pdf) | Primes LLM generation from existing entity graph |
| Financial NER with LLMs | [arXiv 2501.02237](https://arxiv.org/abs/2501.02237) | Domain-specific LLM outperforms encoder models |
| GPT-NER (NAACL 2025) | [ACL Anthology](https://aclanthology.org/2025.findings-naacl.239/) | Self-verification strategy reduces hallucinated entities |

---

## 2. State of the Art — RE (2025–2026)

### LLM-Empowered KG Construction — Survey

**Paper:** [arXiv 2510.20345](https://arxiv.org/abs/2510.20345)

Dominant 2025 pattern: **LLM with structured output schema** as the RE backbone.
LLMs understand implicit relations that span sentences or paragraphs — GLiREL only
operates on within-sentence token spans.

### Towards Practical GraphRAG

**Paper:** [arXiv 2507.03226](https://arxiv.org/abs/2507.03226)

Introduces the **draft-then-verify** pattern:
1. Fast dependency-based RE for first pass (achieves 94% of LLM quality)
2. LLM verification only for triples below a confidence threshold

Directly applicable: run GLiREL / spaCy SVO as draft, use GGUF chat for
verification of uncertain triples.

### Replicating DeepSeek-R1 for Information Extraction

**Blog:** [HuggingFace](https://huggingface.co/blog/Ihor/replicating-deepseek-r1-for-information-extraction)

Fine-tuning an 8B model on ~1 000 RE examples with chain-of-thought supervision
reaches SOTA RE F1. Small SFT dataset is sufficient because the base model already
understands relation semantics from pre-training.

---

## 3. State of the Art — ER (2025–2026)

### In-context Clustering ER with LLMs

**Paper:** [arXiv 2506.02509](https://arxiv.org/html/2506.02509v1)

**LLM-CER** presents a cluster of HNSW-blocked candidates to the LLM and asks
"which of these refer to the same entity?" The LLM makes the merge decision rather
than a fixed cosine + string-similarity threshold.

**Result:** 150% higher accuracy vs prior methods, 5× fewer LLM API calls than
pairwise comparison.

Maps directly onto the existing pipeline:
- HNSW blocking already implemented in `kg_resolve.py` stage 1
- Jaro-Winkler + Leiden stages 2–3 are replaced by LLM cluster judgment

### Progressive ER Design Space

**Paper:** [arXiv 2503.08298](https://arxiv.org/html/2503.08298)

Four-stage framework: **filter → weight → schedule → match**. The scheduling stage
(which candidate pairs to present to the LLM first) dramatically affects the
cost/quality tradeoff. HNSW blocking covers the filter stage perfectly.

### Multi-Agent RAG for ER

**Paper:** [MDPI 2025](https://www.mdpi.com/2073-431X/14/12/525)

Decomposes ER into specialised agents: direct-match agent, transitive-linkage agent,
cluster agent, residential-movement agent. **94.3% accuracy** on name variation
matching, **61% fewer API calls** vs single-LLM baseline.

---

## 4. Open Source KG Frameworks (2025)

| Project | Approach | Relevant Aspect |
|---|---|---|
| [LightRAG](https://github.com/HKUDS/LightRAG) (EMNLP 2025) | LLM-based KG extraction + dual-level retrieval | 10× cheaper than Microsoft GraphRAG; supports Qwen3-30B-A3B locally |
| [Microsoft GraphRAG](https://github.com/microsoft/graphrag) | LLM community summarisation + global search | Leiden communities align with muninn's `graph_leiden` TVF |
| [FalkorDB GraphRAG SDK](https://github.com/FalkorDB/GraphRAG-SDK) | LLM schema-driven KG construction | Production reference: schema → entities → relations pipeline |
| [Neo4j LLM KG Builder](https://neo4j.com/blog/developer/llm-knowledge-graph-builder-release/) | LLM-driven NER/RE into graph DB | Production pattern using OpenAI function calling |
| [GLiNER2](https://github.com/fastino-ai/GLiNER2) | Single model: NER + RE + classify + structured extract (205M/340M) | Replaces GLiNER + GLiREL + spaCy with one `pip install gliner2` |
| [instructor](https://python.useinstructor.com/integrations/llama-cpp-python/) | Pydantic-typed structured output via llama-cpp-python GBNF | Python adapter layer for GGUF models |

**Key differentiator for this project:** LightRAG, GraphRAG, and Neo4j all require
external graph databases. This project stores everything in SQLite — uniquely
valuable for **embedded / offline / edge deployment** where external infra is
unavailable.

---

## 5. GGUF Chat Model Catalog

### Recommended Models for NER / RE / ER Tasks

| Name | Params | GGUF (Q4\_K\_M) | Context | Strengths | Task fit |
|---|---|---|---|---|---|
| **Qwen3-4B-Instruct** | 4B | ~2.7 GB | 32k | Strong JSON discipline, thinking mode, multilingual | NER draft pass, fast RE |
| **Qwen3-8B-Instruct** | 8B | ~5.1 GB | 32k | Best small-model JSON; thinking mode for ambiguous spans | NER + RE production pass |
| **DeepSeek-R1-0528-Qwen3-8B** | 8B | ~5.1 GB | 32k | CoT + F1 > 0.95 on bio-NLP NER; Qwen3 base | High-quality NER/RE verification |
| **Phi-4-mini-Instruct** | 3.8B | ~2.5 GB | 16k | Microsoft; structured reasoning | ER judgment (lighter) |
| **Phi-4** | 14B | ~8.8 GB | 16k | Excellent structured reasoning | ER cluster judgment |
| **Qwen3-14B-Instruct** | 14B | ~9.0 GB | 32k | Thinking mode at 14B; very strong | Community summarisation, ER |
| **Llama-3.1-8B-Instruct** | 8B | ~5.0 GB | 128k | Well-tested, long context | Long-document RE |
| **Llama-3.3-70B-Instruct** | 70B | ~42 GB | 128k | Near-GPT-4 quality | Reference benchmark comparison |

> **Quantisation note:** Q4\_K\_M is the recommended sweet-spot — comparable
> quality to Q8\_0 on extraction tasks while using ~40% less memory.
> Verify HuggingFace URLs before adding to the catalog (official Qwen GGUFs at
> `Qwen/Qwen3-{4,8,14,32}B-GGUF`; Llama at `bartowski/Meta-Llama-3.1-8B-Instruct-GGUF`).

### Embedding Models (existing catalog)

| Name | Params | Dim | Already in prep gguf? |
|---|---|---|---|
| MiniLM | 22M | 384 | ✓ |
| NomicEmbed | 137M | 768 | ✓ |
| BGE-Large | 335M | 1024 | ✓ |
| Qwen3-Embedding-8B | 8B | 4096 | ✗ (see `docs/plans/gguf_ml_integration.md`) |

---

## 6. Additions Required for `benchmarks.harness prep gguf`

### New: `GGUF_CHAT_MODELS` catalog

The existing `gguf_models.py` only covers embedding models. Chat (completion) models
need a parallel catalog with different metadata:

```
gguf_models.py additions:
  GGUF_CHAT_MODELS: list[dict[str, str]]  — chat model definitions
  ChatModelPrepTask                        — same download pattern, adds ctx_len / task fields
  GGUF_CHAT_PREP_TASKS                    — list[PrepTask]
```

Each chat model entry needs these extra fields versus the embedding catalog:

| Field | Example | Purpose |
|---|---|---|
| `task` | `"chat"` | Distinguishes from embedding models |
| `ctx_len` | `"32768"` | Max context window |
| `chat_template` | `"chatml"` | Template format for the prompt |

### New CLI target: `prep gguf-chat`

Mirrors the existing `prep gguf` command:

```bash
uv run -m benchmarks.harness prep gguf-chat           # download all chat models
uv run -m benchmarks.harness prep gguf-chat --model Qwen3-8B
uv run -m benchmarks.harness prep gguf-chat --status
```

### Phase 1: Minimum viable catalog (NER/RE tasks)

Start with two models covering the most common hardware profiles:

| Priority | Model | Reason |
|---|---|---|
| 1 | Qwen3-4B-Instruct (Q4\_K\_M) | Fits in 4 GB VRAM; strong JSON; useful on laptops |
| 2 | Qwen3-8B-Instruct (Q4\_K\_M) | Best quality/size for structured extraction |

### Phase 2: Extended catalog (ER + community summarisation)

| Priority | Model | Reason |
|---|---|---|
| 3 | Phi-4 (Q4\_K\_M, 14B) | ER cluster judgment quality |
| 4 | Llama-3.1-8B-Instruct (Q4\_K\_M) | Comparison baseline; 128k context |

---

## 7. Cross-Cutting Takeaways

1. **Hybrid encoder + LLM is the recommended pattern.** Encoder models (GLiNER
   family) for bulk fast extraction; LLM for disambiguation, cross-sentence RE, and
   ER cluster judgment on uncertain pairs.

2. **GBNF grammar-constrained decoding** (built into llama.cpp) eliminates
   hallucinated JSON format errors — forces 100% valid structured output at no
   quality cost.

3. **Adapter ABC pattern is the correct seam.** `NerModelAdapter` and
   `ReModelAdapter` in `kg_ner_adapters.py` / `kg_re_adapters.py` allow
   `LlmNerAdapter` and `LlmREAdapter` to plug in with zero changes to the harness,
   benchmark registry, or chart code.

4. **The 97-permutation benchmark suite runs unchanged** — new LLM adapters add
   permutations, they do not modify existing ones.

5. **Community summarisation** is a new capability with no existing counterpart:
   present the entity list and relations for a Leiden community to a chat model →
   receive a human-readable community name / summary. Directly usable in the viz
   KG Pipeline Explorer.
