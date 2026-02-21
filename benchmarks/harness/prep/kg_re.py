"""Prep module: download relation extraction benchmark datasets.

Downloads and normalizes RE datasets to a common JSONL format
for triple-level F1 evaluation.

Datasets:
    DocRED  — HuggingFace `docred` (document-level RE)
    WebNLG  — HuggingFace `web_nlg` (RDF triples)
    TACRED  — HuggingFace `DFKI-SLT/tacred` (sentence-level RE)

Output format per dataset in KG_DIR/re/{name}/:
    texts.jsonl   — {"id": int, "text": str}
    triples.jsonl — {"text_id": int, "subject": str, "predicate": str, "object": str}

Source: docs/plans/ner_extraction_models_and_datasets.md
"""

import logging

from benchmarks.harness.common import KG_DIR
from benchmarks.harness.prep.common import count_jsonl_lines, write_jsonl

log = logging.getLogger(__name__)

KNOWN_RE_DATASETS = ["docred", "webnlg", "tacred"]


def print_status():
    """Print status of RE benchmark datasets."""
    re_dir = KG_DIR / "re"

    print("=== RE Dataset Status ===\n")
    print(f"  {'DATASET':<16s}   {'DIR':<32s}   {'STATUS'}")
    print(f"  {'-' * 16}   {'-' * 32}   {'-' * 12}")

    for ds_name in KNOWN_RE_DATASETS:
        ds_dir = re_dir / ds_name
        texts_path = ds_dir / "texts.jsonl"
        triples_path = ds_dir / "triples.jsonl"

        if texts_path.exists() and triples_path.exists():
            n_texts = count_jsonl_lines(texts_path)
            n_triples = count_jsonl_lines(triples_path)
            print(
                f"  {ds_name:<16s}   {'re/' + ds_name + '/':<32s}   READY ({n_texts} texts, {n_triples} triples)"
            )
        elif ds_dir.exists():
            print(f"  {ds_name:<16s}   {'re/' + ds_name + '/':<32s}   INCOMPLETE")
        else:
            print(f"  {ds_name:<16s}   {'re/' + ds_name + '/':<32s}   MISSING")

    # Show any extra datasets
    if re_dir.exists():
        for ds_dir in sorted(re_dir.iterdir()):
            if ds_dir.is_dir() and ds_dir.name not in KNOWN_RE_DATASETS:
                has_texts = (ds_dir / "texts.jsonl").exists()
                status = "READY" if has_texts else "INCOMPLETE"
                print(f"  {ds_dir.name:<16s}   {'re/' + ds_dir.name + '/':<32s}   {status} (extra)")

    print()


def prep_re_datasets(dataset=None, status_only=False, force=False):
    """Download and normalize RE benchmark datasets.

    Args:
        dataset: Specific dataset (docred, webnlg, tacred). If None, downloads all.
        status_only: If True, show status and return.
        force: If True, re-download datasets even if they exist.
    """
    if status_only:
        print_status()
        return

    re_dir = KG_DIR / "re"
    re_dir.mkdir(parents=True, exist_ok=True)

    datasets_to_prep = [dataset] if dataset else KNOWN_RE_DATASETS

    for ds_name in datasets_to_prep:
        ds_dir = re_dir / ds_name

        if ds_dir.exists() and (ds_dir / "texts.jsonl").exists() and not force:
            log.info("  Dataset %s: already exists (use --force to re-download)", ds_name)
            continue

        ds_dir.mkdir(parents=True, exist_ok=True)

        if ds_name == "docred":
            _prep_docred(ds_dir)
        elif ds_name == "webnlg":
            _prep_webnlg(ds_dir)
        elif ds_name == "tacred":
            _prep_tacred(ds_dir)
        else:
            log.warning("  Dataset %s: download not yet implemented", ds_name)

    log.info("RE dataset prep complete. Datasets in %s", re_dir)


def _write_dataset(output_dir, texts, triples):
    """Write normalized texts and triples to JSONL files."""
    write_jsonl(output_dir / "texts.jsonl", texts)
    write_jsonl(output_dir / "triples.jsonl", triples)
    log.info("  Wrote %d texts and %d triples to %s", len(texts), len(triples), output_dir)


def _prep_docred(output_dir):
    """Download DocRED and extract document-level relation triples."""
    try:
        from datasets import load_dataset
    except ImportError:
        log.warning("  'datasets' package not available — skipping docred")
        log.warning("  Install with: uv pip install datasets")
        return

    log.info("  Downloading DocRED from HuggingFace...")
    ds = load_dataset("docred", split="validation")

    texts = []
    triples = []

    for i, row in enumerate(ds):
        # DocRED has sents (list of sentence token lists) and labels
        sents = row.get("sents", [])
        text = " ".join(" ".join(sent) for sent in sents)
        texts.append({"id": i, "text": text})

        # Extract triples from labels + vertex_set
        vertex_set = row.get("vertexSet", [])
        labels = row.get("labels", [])

        for label in labels:
            head_idx = label.get("head", label.get("h", -1))
            tail_idx = label.get("tail", label.get("t", -1))
            relation = label.get("relation_text", label.get("r", "unknown"))

            if 0 <= head_idx < len(vertex_set) and 0 <= tail_idx < len(vertex_set):
                head_name = vertex_set[head_idx][0].get("name", "?") if vertex_set[head_idx] else "?"
                tail_name = vertex_set[tail_idx][0].get("name", "?") if vertex_set[tail_idx] else "?"
                triples.append(
                    {
                        "text_id": i,
                        "subject": head_name,
                        "predicate": str(relation),
                        "object": tail_name,
                    }
                )

    _write_dataset(output_dir, texts, triples)


def _prep_webnlg(output_dir):
    """Download WebNLG and extract RDF triples."""
    try:
        from datasets import load_dataset
    except ImportError:
        log.warning("  'datasets' package not available — skipping webnlg")
        log.warning("  Install with: uv pip install datasets")
        return

    log.info("  Downloading WebNLG from HuggingFace...")
    ds = load_dataset("web_nlg", "release_v3.0_en", split="test")

    texts = []
    triples = []

    for i, row in enumerate(ds):
        # WebNLG has lex (text realizations) and modified_triple_sets
        lex_entries = row.get("lex", {}).get("text", [])
        text = lex_entries[0] if lex_entries else ""
        texts.append({"id": i, "text": text})

        # Extract triples from modified_triple_sets
        triple_sets = row.get("modified_triple_sets", {}).get("mtriple_set", [])
        for triple_set in triple_sets:
            for triple_str in triple_set:
                parts = triple_str.split(" | ")
                if len(parts) == 3:
                    triples.append(
                        {
                            "text_id": i,
                            "subject": parts[0].strip(),
                            "predicate": parts[1].strip(),
                            "object": parts[2].strip(),
                        }
                    )

    _write_dataset(output_dir, texts, triples)


def _prep_tacred(output_dir):
    """Download TACRED and extract sentence-level relation triples."""
    try:
        from datasets import load_dataset
    except ImportError:
        log.warning("  'datasets' package not available — skipping tacred")
        log.warning("  Install with: uv pip install datasets")
        return

    log.info("  Downloading TACRED from HuggingFace...")
    try:
        ds = load_dataset("DFKI-SLT/tacred", split="test")
    except Exception as exc:
        log.warning("  Failed to load TACRED: %s", exc)
        log.warning("  TACRED may require accepting a license on HuggingFace")
        return

    texts = []
    triples = []

    for i, row in enumerate(ds):
        tokens = row.get("tokens", row.get("token", []))
        text = " ".join(tokens)
        texts.append({"id": i, "text": text})

        relation = row.get("relation", "no_relation")
        if relation != "no_relation":
            subj_start = row.get("subj_start", 0)
            subj_end = row.get("subj_end", 0)
            obj_start = row.get("obj_start", 0)
            obj_end = row.get("obj_end", 0)

            subject = " ".join(tokens[subj_start : subj_end + 1]) if subj_end >= subj_start else "?"
            obj = " ".join(tokens[obj_start : obj_end + 1]) if obj_end >= obj_start else "?"

            triples.append(
                {
                    "text_id": i,
                    "subject": subject,
                    "predicate": relation,
                    "object": obj,
                }
            )

    _write_dataset(output_dir, texts, triples)
