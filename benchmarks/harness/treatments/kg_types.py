"""Shared types for KG benchmark treatments.

Contains EntityMention and NerModelAdapter ABC, shared between kg_extract.py
and kg_ner_adapters.py. Extracted to avoid circular imports.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class EntityMention:
    """A single entity mention extracted from text."""

    text: str
    label: str
    start: int
    end: int
    score: float = 1.0


class NerModelAdapter(ABC):
    """Common interface for all NER extraction models."""

    @abstractmethod
    def load(self) -> None:
        """Load the model into memory."""

    @abstractmethod
    def extract(self, text: str, labels: list[str]) -> list[EntityMention]:
        """Extract entity mentions from text."""

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Model identifier string."""

    @property
    @abstractmethod
    def model_type(self) -> str:
        """Model family: 'gliner', 'gner', 'spacy', 'fts5', 'nuner'."""
