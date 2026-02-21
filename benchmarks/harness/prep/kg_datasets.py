"""Prep module: download KG benchmark datasets (NER, RE, ER).

Unified replacement for kg_ner.py, kg_re.py, kg_er.py.
Uses HFDatasetPrepTask for HuggingFace datasets, CrossNERPrepTask for
CrossNER (direct GitHub download), and FebrlPrepTask for recordlinkage
Febrl datasets.

HFDatasetPrepTask delegates caching entirely to the HuggingFace `datasets`
library — we probe the cache rather than checking file paths directly.

CrossNERPrepTask bypasses the broken HF dataset script by downloading raw
BIO-tagged text files directly from the CrossNER GitHub repository.
"""

import json
import logging
import urllib.request
from pathlib import Path

from benchmarks.harness.common import KG_DIR
from benchmarks.harness.prep.base import PrepTask
from benchmarks.harness.prep.common import bio_to_spans

log = logging.getLogger(__name__)


class CrossNERPrepTask(PrepTask):
    """Download CrossNER domain data directly from GitHub (bypasses broken HF script).

    CrossNER's raw data lives on GitHub in a simple tab-separated BIO format:
    ``token\\ttag`` per line, blank lines separate sentences. We download these
    directly and convert to the JSONL format downstream treatments expect.
    """

    DOMAINS = ("ai", "conll2003", "literature", "music", "politics", "science")
    SPLITS = ("train", "dev", "test")
    _BASE_URL = "https://raw.githubusercontent.com/zliucr/CrossNER/main/ner_data"

    def __init__(self, domain: str):
        if domain not in self.DOMAINS:
            raise ValueError(f"Unknown CrossNER domain: {domain!r} (expected one of {self.DOMAINS})")
        self._domain = domain

    @property
    def task_id(self) -> str:
        return f"crossner:{self._domain}"

    @property
    def label(self) -> str:
        return f"CrossNER {self._domain}"

    def outputs(self) -> list[Path]:
        out_dir = KG_DIR / "ner" / f"crossner_{self._domain}"
        return [out_dir / "texts.jsonl", out_dir / "entities.jsonl"]

    def fetch(self, force: bool = False) -> None:
        raw_dir = KG_DIR / "ner" / f"crossner_{self._domain}" / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)

        for split in self.SPLITS:
            url = f"{self._BASE_URL}/{self._domain}/{split}.txt"
            dest = raw_dir / f"{split}.txt"
            if dest.exists() and not force:
                log.info("  %s/%s.txt: cached (skip)", self.task_id, split)
                continue
            log.info("  Downloading %s/%s.txt ...", self.task_id, split)
            urllib.request.urlretrieve(url, dest)

    def transform(self) -> None:
        raw_dir = KG_DIR / "ner" / f"crossner_{self._domain}" / "raw"
        out_dir = KG_DIR / "ner" / f"crossner_{self._domain}"

        all_texts: list[dict] = []
        all_entities: list[dict] = []
        sent_id = 0

        for split in self.SPLITS:
            raw_path = raw_dir / f"{split}.txt"
            if not raw_path.exists():
                log.warning("  %s: missing raw/%s.txt — skipping", self.task_id, split)
                continue

            tokens: list[str] = []
            tags: list[str] = []

            for line in raw_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    # Sentence boundary
                    if tokens:
                        text = " ".join(tokens)
                        spans = bio_to_spans(tokens, tags)
                        all_texts.append({"id": sent_id, "split": split, "text": text})
                        for span in spans:
                            all_entities.append({"text_id": sent_id, **span})
                        sent_id += 1
                        tokens = []
                        tags = []
                    continue
                parts = line.split("\t")
                if len(parts) == 2:
                    tokens.append(parts[0])
                    tags.append(parts[1])

            # Flush last sentence (file may not end with blank line)
            if tokens:
                text = " ".join(tokens)
                spans = bio_to_spans(tokens, tags)
                all_texts.append({"id": sent_id, "split": split, "text": text})
                for span in spans:
                    all_entities.append({"text_id": sent_id, **span})
                sent_id += 1

        # Write JSONL outputs
        texts_path = out_dir / "texts.jsonl"
        entities_path = out_dir / "entities.jsonl"
        texts_path.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in all_texts) + "\n",
            encoding="utf-8",
        )
        entities_path.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in all_entities) + "\n",
            encoding="utf-8",
        )
        log.info(
            "  %s: %d sentences, %d entities → %s",
            self.task_id,
            len(all_texts),
            len(all_entities),
            out_dir,
        )


class HFDatasetPrepTask(PrepTask):
    """PrepTask for any HuggingFace dataset."""

    def __init__(self, hf_path: str, hf_config: str | None = None, revision: str | None = None):
        self._hf_path = hf_path
        self._hf_config = hf_config
        self._revision = revision

    @property
    def task_id(self) -> str:
        if self._hf_config:
            return f"{self._hf_path}:{self._hf_config}"
        return self._hf_path

    @property
    def label(self) -> str:
        name = self._hf_path.split("/")[-1]
        if self._hf_config:
            return f"HF {name} ({self._hf_config})"
        return f"HF {name}"

    def outputs(self) -> list:
        # HF cache is opaque — we probe it via status() instead
        return []

    def status(self) -> str:
        """Probe HF cache to determine if dataset is already downloaded."""
        from datasets import load_dataset

        try:
            load_dataset(
                self._hf_path, self._hf_config, revision=self._revision, download_mode="reuse_cache_only"
            )
            return "READY"
        except Exception:
            return "MISSING"

    def fetch(self, force: bool = False) -> None:
        from datasets import load_dataset

        mode = "force_redownload" if force else None
        log.info("  Downloading %s ...", self.task_id)
        load_dataset(self._hf_path, self._hf_config, revision=self._revision, download_mode=mode)
        log.info("  %s: cached in HF cache", self.task_id)


class FebrlPrepTask(PrepTask):
    """PrepTask for Febrl entity resolution datasets via recordlinkage."""

    def __init__(self, name: str):
        self._name = name

    @property
    def task_id(self) -> str:
        return self._name

    @property
    def label(self) -> str:
        return f"Febrl {self._name}"

    def outputs(self) -> list:
        out_dir = KG_DIR / "er" / self._name
        if self._name == "febrl4":
            return [out_dir / "febrl4_a.parquet", out_dir / "febrl4_b.parquet"]
        return [out_dir / f"{self._name}.parquet"]

    def fetch(self, force: bool = False) -> None:
        from recordlinkage.datasets import load_febrl1, load_febrl4

        out_dir = KG_DIR / "er" / self._name
        out_dir.mkdir(parents=True, exist_ok=True)

        if self._name == "febrl1":
            df = load_febrl1()
            out_path = out_dir / "febrl1.parquet"
            df.to_parquet(out_path)
            log.info("  %s: saved %d records to %s", self._name, len(df), out_path)
        elif self._name == "febrl4":
            df_a, df_b = load_febrl4()
            path_a = out_dir / "febrl4_a.parquet"
            path_b = out_dir / "febrl4_b.parquet"
            df_a.to_parquet(path_a)
            df_b.to_parquet(path_b)
            log.info("  %s: saved %d + %d records to %s", self._name, len(df_a), len(df_b), out_dir)
        else:
            log.warning("  Unknown Febrl dataset: %s", self._name)


# ── Task registry ────────────────────────────────────────────────

KG_PREP_TASKS: list[PrepTask] = [
    # NER — CrossNER (direct GitHub download, bypasses broken HF script)
    CrossNERPrepTask("ai"),
    CrossNERPrepTask("conll2003"),
    CrossNERPrepTask("literature"),
    CrossNERPrepTask("music"),
    CrossNERPrepTask("politics"),
    CrossNERPrepTask("science"),
    # NER — Few-NERD (works with standard HF loading)
    HFDatasetPrepTask("DFKI-SLT/few-nerd", "supervised"),
    HFDatasetPrepTask("DFKI-SLT/few-nerd", "inter"),
    HFDatasetPrepTask("DFKI-SLT/few-nerd", "intra"),
    # RE (parquet branch for script-based datasets)
    HFDatasetPrepTask("thunlp/docred", revision="refs/convert/parquet"),
    HFDatasetPrepTask("webnlg-challenge/web_nlg", revision="refs/convert/parquet"),
    HFDatasetPrepTask("DFKI-SLT/conll04"),
    # ER
    FebrlPrepTask("febrl1"),
    FebrlPrepTask("febrl4"),
]


def print_status():
    """Print status of all KG dataset prep tasks."""
    print("=== KG Dataset Status ===\n")
    print(f"  {'TASK_ID':<40s}   {'LABEL':<28s}   {'STATUS'}")
    print(f"  {'-' * 40}   {'-' * 28}   {'-' * 8}")

    for task in KG_PREP_TASKS:
        st = task.status()
        print(f"  {task.task_id:<40s}   {task.label:<28s}   {st}")
    print()


def prep_kg_datasets(dataset: str | None = None, status_only: bool = False, force: bool = False):
    """Download KG benchmark datasets (NER, RE, ER).

    Args:
        dataset: Specific task_id to download. If None, downloads all.
        status_only: If True, show status and return.
        force: If True, re-download datasets even if cached.
    """
    if status_only:
        print_status()
        return

    tasks = KG_PREP_TASKS
    if dataset:
        tasks = [t for t in KG_PREP_TASKS if t.task_id == dataset]
        if not tasks:
            log.error("Unknown KG dataset: %s", dataset)
            log.info("Available: %s", ", ".join(t.task_id for t in KG_PREP_TASKS))
            return

    log.info("Downloading %d KG dataset(s)...", len(tasks))
    for task in tasks:
        task.run(force=force)
    log.info("KG dataset prep complete.")
