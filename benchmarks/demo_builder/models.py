"""ModelPool: pre-loads and holds shared ML models for the build pipeline.

This module is imported inside _cmd_build() in cli.py, so top-level ML
imports are acceptable — they are only loaded when a build is requested.
"""

from __future__ import annotations

import logging
from typing import Any

import spacy
from gliner import GLiNER
from glirel import GLiREL
from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer

from benchmarks.demo_builder.constants import EMBEDDING_MODELS

log = logging.getLogger(__name__)


class ModelPool:
    """Holds pre-loaded ML models shared across all build permutations.

    Constructed once in _cmd_build(), passed to each DemoBuild instance.
    """

    def __init__(self, model_names: list[str]) -> None:
        self._model_names = model_names
        self._ner_model: GLiNER | None = None
        self._re_model: GLiREL | None = None
        self._nlp: spacy.language.Language | None = None
        self._st_models: dict[str, SentenceTransformer] = {}

    def load_all(self) -> None:
        """Load all required ML models. Call once before starting builds."""
        log.info("Loading ML models...")

        log.info("  GLiNER medium-v2.1...")
        self._ner_model = GLiNER.from_pretrained("urchade/gliner_medium-v2.1")

        log.info("  GLiREL large-v0...")
        glirel_dir = snapshot_download("jackboyla/glirel-large-v0")
        self._re_model = GLiREL._from_pretrained(
            model_id=glirel_dir,
            revision=None,
            cache_dir=None,
            force_download=False,
            proxies=None,
            resume_download=False,
            local_files_only=True,
            token=None,
        )

        log.info("  spaCy en_core_web_lg...")
        self._nlp = spacy.load("en_core_web_lg")

        for mname in sorted(self._model_names):
            model_info = EMBEDDING_MODELS[mname]
            log.info("  SentenceTransformer %s...", model_info["st_name"])
            st_kwargs: dict[str, Any] = {}
            if model_info.get("trust_remote_code"):
                st_kwargs["trust_remote_code"] = True
            self._st_models[mname] = SentenceTransformer(model_info["st_name"], **st_kwargs)

        log.info("All models loaded")

    @property
    def ner_model(self) -> Any:
        """Pre-loaded GLiNER NER model."""
        return self._ner_model

    @property
    def re_model(self) -> Any:
        """Pre-loaded GLiREL relation extraction model."""
        return self._re_model

    @property
    def nlp(self) -> Any:
        """Pre-loaded spaCy language pipeline."""
        return self._nlp

    def st_model(self, name: str) -> Any:
        """Get a pre-loaded SentenceTransformer by model short name."""
        return self._st_models[name]
