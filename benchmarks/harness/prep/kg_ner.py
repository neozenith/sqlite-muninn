"""Prep module: download NER benchmark datasets.

Downloads and normalizes NER benchmark datasets to a common JSONL format
for entity-level micro F1 evaluation.

Datasets:
    CoNLL-2003  — HuggingFace `eriktks/conll2003` (BIO tags → entity spans)
    CrossNER    — HuggingFace `zeroshot/crossner` (BIO tags → entity spans)
    Few-NERD    — HuggingFace `DFKI-SLT/few-nerd` (fine-grained entity spans)

Output format per dataset in KG_DIR/ner/{name}/:
    texts.jsonl    — {"id": int, "text": str, "tokens": list[str]}
    entities.jsonl — {"text_id": int, "start": int, "end": int, "label": str, "surface": str}

Source: docs/plans/ner_extraction_models_and_datasets.md
"""

import logging

from benchmarks.harness.common import KG_DIR
from benchmarks.harness.prep.common import count_jsonl_lines, write_jsonl

log = logging.getLogger(__name__)

KNOWN_NER_DATASETS = ["conll2003", "crossner", "fewnerd"]


def print_status():
    """Print status of NER benchmark datasets."""
    ner_dir = KG_DIR / "ner"

    print("=== NER Dataset Status ===\n")
    print(f"  {'DATASET':<16s}   {'DIR':<32s}   {'STATUS'}")
    print(f"  {'-' * 16}   {'-' * 32}   {'-' * 12}")

    for ds_name in KNOWN_NER_DATASETS:
        ds_dir = ner_dir / ds_name
        texts_path = ds_dir / "texts.jsonl"
        entities_path = ds_dir / "entities.jsonl"

        if texts_path.exists() and entities_path.exists():
            n_texts = count_jsonl_lines(texts_path)
            n_ents = count_jsonl_lines(entities_path)
            print(
                f"  {ds_name:<16s}   {'ner/' + ds_name + '/':<32s}   READY ({n_texts} texts, {n_ents} entities)"
            )
        elif ds_dir.exists():
            print(f"  {ds_name:<16s}   {'ner/' + ds_name + '/':<32s}   INCOMPLETE")
        else:
            print(f"  {ds_name:<16s}   {'ner/' + ds_name + '/':<32s}   MISSING")

    # Show any extra datasets
    if ner_dir.exists():
        for ds_dir in sorted(ner_dir.iterdir()):
            if ds_dir.is_dir() and ds_dir.name not in KNOWN_NER_DATASETS:
                has_texts = (ds_dir / "texts.jsonl").exists()
                status = "READY" if has_texts else "INCOMPLETE"
                print(f"  {ds_dir.name:<16s}   {'ner/' + ds_dir.name + '/':<32s}   {status} (extra)")

    print()


def prep_ner_datasets(dataset=None, status_only=False, force=False):
    """Download and normalize NER benchmark datasets.

    Args:
        dataset: Specific dataset (conll2003, crossner, fewnerd). If None, downloads all.
        status_only: If True, show status and return.
        force: If True, re-download datasets even if they exist.
    """
    if status_only:
        print_status()
        return

    ner_dir = KG_DIR / "ner"
    ner_dir.mkdir(parents=True, exist_ok=True)

    datasets_to_prep = [dataset] if dataset else KNOWN_NER_DATASETS

    for ds_name in datasets_to_prep:
        ds_dir = ner_dir / ds_name

        if ds_dir.exists() and (ds_dir / "texts.jsonl").exists() and not force:
            log.info("  Dataset %s: already exists (use --force to re-download)", ds_name)
            continue

        ds_dir.mkdir(parents=True, exist_ok=True)

        if ds_name == "conll2003":
            _prep_conll2003(ds_dir)
        elif ds_name == "crossner":
            _prep_crossner(ds_dir)
        elif ds_name == "fewnerd":
            _prep_fewnerd(ds_dir)
        else:
            log.warning("  Dataset %s: download not yet implemented", ds_name)

    log.info("NER dataset prep complete. Datasets in %s", ner_dir)


def bio_to_spans(tokens: list[str], tags: list[str]) -> list[dict]:
    """Convert BIO-tagged tokens to character-offset entity spans.

    Args:
        tokens: List of word tokens.
        tags: List of BIO tags (B-PER, I-PER, O, etc.).

    Returns:
        List of {"start": int, "end": int, "label": str, "surface": str} dicts.
    """
    spans = []
    current_start = None
    current_label = None
    char_offset = 0
    token_starts = []

    # Compute character offsets for each token (space-separated)
    for token in tokens:
        token_starts.append(char_offset)
        char_offset += len(token) + 1  # +1 for space

    for i, (token, tag) in enumerate(zip(tokens, tags)):
        if tag.startswith("B-"):
            # Close previous span if open
            if current_start is not None:
                surface = " ".join(tokens[current_start:i])
                spans.append(
                    {
                        "start": token_starts[current_start],
                        "end": token_starts[i - 1] + len(tokens[i - 1]),
                        "label": current_label,
                        "surface": surface,
                    }
                )
            current_start = i
            current_label = tag[2:]
        elif tag.startswith("I-"):
            # Continue current span (if label matches)
            if current_start is None or tag[2:] != current_label:
                # Malformed BIO: I without matching B — treat as B
                if current_start is not None:
                    surface = " ".join(tokens[current_start:i])
                    spans.append(
                        {
                            "start": token_starts[current_start],
                            "end": token_starts[i - 1] + len(tokens[i - 1]),
                            "label": current_label,
                            "surface": surface,
                        }
                    )
                current_start = i
                current_label = tag[2:]
        else:
            # O tag — close current span
            if current_start is not None:
                surface = " ".join(tokens[current_start:i])
                spans.append(
                    {
                        "start": token_starts[current_start],
                        "end": token_starts[i - 1] + len(tokens[i - 1]),
                        "label": current_label,
                        "surface": surface,
                    }
                )
                current_start = None
                current_label = None

    # Close trailing span
    if current_start is not None:
        surface = " ".join(tokens[current_start:])
        spans.append(
            {
                "start": token_starts[current_start],
                "end": token_starts[-1] + len(tokens[-1]),
                "label": current_label,
                "surface": surface,
            }
        )

    return spans


def _write_dataset(output_dir, texts, entities):
    """Write normalized texts and entities to JSONL files."""
    write_jsonl(output_dir / "texts.jsonl", texts)
    write_jsonl(output_dir / "entities.jsonl", entities)
    log.info("  Wrote %d texts and %d entities to %s", len(texts), len(entities), output_dir)


def _prep_conll2003(output_dir):
    """Download CoNLL-2003 via HuggingFace datasets and convert BIO to spans."""
    try:
        from datasets import load_dataset
    except ImportError:
        log.warning("  'datasets' package not available — skipping conll2003")
        log.warning("  Install with: uv pip install datasets")
        return

    log.info("  Downloading CoNLL-2003 from HuggingFace...")
    ds = load_dataset("eriktks/conll2003", split="test")

    # CoNLL-2003 tag mapping
    ner_tags = ds.features["ner_tags"].feature.names

    texts = []
    entities = []

    for i, row in enumerate(ds):
        tokens = row["tokens"]
        tags = [ner_tags[t] for t in row["ner_tags"]]
        text = " ".join(tokens)

        texts.append({"id": i, "text": text, "tokens": tokens})

        spans = bio_to_spans(tokens, tags)
        for span in spans:
            entities.append({"text_id": i, **span})

    _write_dataset(output_dir, texts, entities)


def _prep_crossner(output_dir):
    """Download CrossNER via HuggingFace and convert BIO to spans."""
    try:
        from datasets import load_dataset
    except ImportError:
        log.warning("  'datasets' package not available — skipping crossner")
        log.warning("  Install with: uv pip install datasets")
        return

    log.info("  Downloading CrossNER from HuggingFace...")
    ds = load_dataset("zeroshot/crossner", split="test")

    texts = []
    entities = []

    for i, row in enumerate(ds):
        tokens = row["tokens"]
        tags = row["ner_tags"]
        # CrossNER ner_tags may be string labels or ints — normalize
        if tags and isinstance(tags[0], int):
            tag_names = ds.features["ner_tags"].feature.names
            tags = [tag_names[t] for t in tags]

        text = " ".join(tokens)
        texts.append({"id": i, "text": text, "tokens": tokens})

        spans = bio_to_spans(tokens, tags)
        for span in spans:
            entities.append({"text_id": i, **span})

    _write_dataset(output_dir, texts, entities)


def _prep_fewnerd(output_dir):
    """Download Few-NERD via HuggingFace and convert to spans."""
    try:
        from datasets import load_dataset
    except ImportError:
        log.warning("  'datasets' package not available — skipping fewnerd")
        log.warning("  Install with: uv pip install datasets")
        return

    log.info("  Downloading Few-NERD from HuggingFace...")
    ds = load_dataset("DFKI-SLT/few-nerd", "supervised", split="test")

    texts = []
    entities = []

    for i, row in enumerate(ds):
        tokens = row["tokens"]
        tags = row["ner_tags"]
        # Few-NERD uses fine-grained tags — may be ints or strings
        if tags and isinstance(tags[0], int):
            tag_names = ds.features["ner_tags"].feature.names
            tags = [tag_names[t] for t in tags]

        text = " ".join(tokens)
        texts.append({"id": i, "text": text, "tokens": tokens})

        spans = bio_to_spans(tokens, tags)
        for span in spans:
            entities.append({"text_id": i, **span})

    _write_dataset(output_dir, texts, entities)
