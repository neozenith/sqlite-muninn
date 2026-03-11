"""NER model adapters for paid API providers (Anthropic, OpenAI, Gemini).

Each adapter wraps a cloud LLM API behind the NerModelAdapter ABC,
normalizing to the common extract(text, labels) -> list[EntityMention] interface.

Adapters:
- AnthropicNerAdapter: NER via Anthropic Claude API (tool_use structured output)
- OpenAiNerAdapter: NER via OpenAI API (json_schema response format)
- GeminiNerAdapter: NER via Google Gemini API (responseSchema parameter)

Environment variables required:
- ANTHROPIC_API_KEY for AnthropicNerAdapter
- OPENAI_API_KEY for OpenAiNerAdapter
- GOOGLE_API_KEY for GeminiNerAdapter
"""

import json
import logging

import anthropic
import google.generativeai as genai
import openai

from benchmarks.harness.treatments.kg_types import EntityMention, NerModelAdapter

log = logging.getLogger(__name__)

# ── Shared constants ────────────────────────────────────────────

_SYSTEM_PROMPT = "You are a precise NER system. Extract entities of specified types. Respond with JSON."

_NER_SCHEMA = {
    "type": "object",
    "properties": {
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "The entity text as it appears in the source."},
                    "type": {"type": "string", "description": "The entity type label."},
                },
                "required": ["text", "type"],
            },
        }
    },
    "required": ["entities"],
}


def _user_prompt(text: str, labels: list[str]) -> str:
    labels_csv = ", ".join(labels)
    return f"Extract entities of types: {labels_csv}\nText: {text}"


def _parse_entities(raw: dict, text: str) -> list[EntityMention]:
    """Parse a raw JSON dict with an 'entities' key into EntityMention list.

    Finds start/end offsets by locating the entity text in the source string.
    Falls back to case-insensitive search. Skips entities not found in source.
    """
    mentions: list[EntityMention] = []
    for entity in raw.get("entities", []):
        entity_text = entity.get("text", "")
        entity_type = entity.get("type", "")

        if not entity_text or not entity_type:
            continue

        # Locate entity span in original text (same pattern as GNERAdapter)
        start = text.find(entity_text)
        if start == -1:
            start = text.lower().find(entity_text.lower())
        if start == -1:
            continue

        end = start + len(entity_text)
        mentions.append(
            EntityMention(
                text=text[start:end],
                label=entity_type,
                start=start,
                end=end,
                score=1.0,
            )
        )

    return mentions


# ── Anthropic ───────────────────────────────────────────────────


class AnthropicNerAdapter(NerModelAdapter):
    """NER via Anthropic Claude API (tool_use / structured output).

    Requires ANTHROPIC_API_KEY environment variable.
    """

    def __init__(self, model: str = "claude-sonnet-4-6"):
        self._model = model
        self._client: anthropic.Anthropic | None = None

    def load(self):
        log.info("Initializing Anthropic client for model: %s", self._model)
        self._client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env

    def extract(self, text: str, labels: list[str]) -> list[EntityMention]:
        assert self._client is not None, "load() must be called before extract()"

        tool = {
            "name": "extract_entities",
            "description": "Extract named entities from the provided text.",
            "input_schema": _NER_SCHEMA,
        }

        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=1024,
                system=_SYSTEM_PROMPT,
                tools=[tool],
                tool_choice={"type": "tool", "name": "extract_entities"},
                messages=[{"role": "user", "content": _user_prompt(text, labels)}],
            )
        except anthropic.APIError as exc:
            log.warning("Anthropic API error: %s", exc)
            return []

        # Extract the tool_use content block
        for block in response.content:
            if block.type == "tool_use" and block.name == "extract_entities":
                return _parse_entities(block.input, text)

        log.warning("No tool_use block in Anthropic response")
        return []

    @property
    def model_id(self) -> str:
        return self._model

    @property
    def model_type(self) -> str:
        return "anthropic"


# ── OpenAI ──────────────────────────────────────────────────────


class OpenAiNerAdapter(NerModelAdapter):
    """NER via OpenAI API (json_schema response format).

    Requires OPENAI_API_KEY environment variable.
    """

    def __init__(self, model: str = "gpt-4.1-mini"):
        self._model = model
        self._client: openai.OpenAI | None = None

    def load(self):
        log.info("Initializing OpenAI client for model: %s", self._model)
        self._client = openai.OpenAI()  # reads OPENAI_API_KEY from env

    def extract(self, text: str, labels: list[str]) -> list[EntityMention]:
        assert self._client is not None, "load() must be called before extract()"

        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "ner",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "entities": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "text": {"type": "string"},
                                    "type": {"type": "string"},
                                },
                                "required": ["text", "type"],
                                "additionalProperties": False,
                            },
                        }
                    },
                    "required": ["entities"],
                    "additionalProperties": False,
                },
            },
        }

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                temperature=0.0,
                max_tokens=1024,
                response_format=response_format,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": _user_prompt(text, labels)},
                ],
            )
        except openai.APIError as exc:
            log.warning("OpenAI API error: %s", exc)
            return []

        content = response.choices[0].message.content
        try:
            parsed = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            log.warning("OpenAI returned invalid JSON: %.200s", content)
            return []

        return _parse_entities(parsed, text)

    @property
    def model_id(self) -> str:
        return self._model

    @property
    def model_type(self) -> str:
        return "openai"


# ── Gemini ──────────────────────────────────────────────────────


class GeminiNerAdapter(NerModelAdapter):
    """NER via Google Gemini API (responseSchema parameter).

    Requires GOOGLE_API_KEY environment variable.
    """

    def __init__(self, model: str = "gemini-2.0-flash"):
        self._model = model
        self._client: genai.GenerativeModel | None = None

    def load(self):
        log.info("Initializing Gemini client for model: %s", self._model)
        genai.configure()  # reads GOOGLE_API_KEY from env
        self._client = genai.GenerativeModel(self._model)

    def extract(self, text: str, labels: list[str]) -> list[EntityMention]:
        assert self._client is not None, "load() must be called before extract()"

        generation_config = {
            "response_mime_type": "application/json",
            "response_schema": _NER_SCHEMA,
            "temperature": 0.0,
            "max_output_tokens": 1024,
        }

        prompt = f"{_SYSTEM_PROMPT}\n\n{_user_prompt(text, labels)}"

        try:
            response = self._client.generate_content(
                prompt,
                generation_config=generation_config,
            )
        except Exception as exc:
            log.warning("Gemini API error: %s", exc)
            return []

        try:
            parsed = json.loads(response.text)
        except (json.JSONDecodeError, TypeError, ValueError):
            log.warning("Gemini returned invalid JSON: %.200s", response.text)
            return []

        return _parse_entities(parsed, text)

    @property
    def model_id(self) -> str:
        return self._model

    @property
    def model_type(self) -> str:
        return "gemini"
