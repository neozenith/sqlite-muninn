/*
 * llama_constants.h — GBNF grammars and system prompts for extraction functions
 *
 * Centralizes all prompt engineering in one reviewable location.
 * Each extraction mode has supervised (labels provided) and unsupervised
 * (open extraction) variants. The GBNF grammars are shared — the string
 * rule already accepts any text for type/relation values.
 *
 * System prompts include 2 few-shot examples (Tesla/Musk, Microsoft/LinkedIn)
 * that demonstrate the expected JSON output format. These are static and
 * NOT templated with user-provided labels.
 */
#ifndef LLAMA_CONSTANTS_H
#define LLAMA_CONSTANTS_H

/* ──────────────────────────────────────────────────────────────────
 * GBNF Grammars for Structured Extraction
 *
 * These grammars are passed to llama_sampler_init_grammar() to force
 * the model to produce valid JSON at the token generation level.
 * The grammar sampler rejects any token that would violate the
 * grammar, so output is guaranteed well-formed — no post-hoc
 * bracket matching or JSON repair needed.
 *
 * muninn_chat() also accepts user-provided GBNF via its 3rd arg.
 * ────────────────────────────────────────────────────────────── */

/* Shared tail rules: string, number, whitespace */
#define GBNF_COMMON_RULES                                                                                              \
    "string ::= \"\\\"\" [^\"\\\\]* \"\\\"\"\n"                                                                        \
    "number ::= [0-9] (\".\" [0-9]+)?\n"                                                                               \
    "ws ::= [ \\t\\n]*\n"

/* NER: {"entities":[...]} or bare array [{...},...] (normalized in result_json_output) */
static const char *GBNF_NER =
    "root ::= (\"{\" ws \"\\\"entities\\\"\" ws \":\" ws \"[\" ws entities ws \"]\" ws \"}\") "
    "| (\"[\" ws entities ws \"]\")\n"
    "entities ::= entity (\",\" ws entity)* | \"\"\n"
    "entity ::= \"{\" ws \"\\\"text\\\"\" ws \":\" ws string ws \",\" ws "
    "\"\\\"type\\\"\" ws \":\" ws string ws \",\" ws "
    "\"\\\"score\\\"\" ws \":\" ws number ws \"}\"\n" GBNF_COMMON_RULES;

/* RE: {"relations":[...]} or bare array [{...},...] (normalized in result_json_output) */
static const char *GBNF_RE =
    "root ::= (\"{\" ws \"\\\"relations\\\"\" ws \":\" ws \"[\" ws relations ws \"]\" ws \"}\") "
    "| (\"[\" ws relations ws \"]\")\n"
    "relations ::= relation (\",\" ws relation)* | \"\"\n"
    "relation ::= \"{\" ws \"\\\"head\\\"\" ws \":\" ws string ws \",\" ws "
    "\"\\\"rel\\\"\" ws \":\" ws string ws \",\" ws "
    "\"\\\"tail\\\"\" ws \":\" ws string ws \",\" ws "
    "\"\\\"score\\\"\" ws \":\" ws number ws \"}\"\n" GBNF_COMMON_RULES;

/* Combined NER+RE: {"entities":[...],"relations":[...]} */
static const char *GBNF_NER_RE =
    "root ::= \"{\" ws \"\\\"entities\\\"\" ws \":\" ws \"[\" ws entities ws \"]\" ws \",\" ws "
    "\"\\\"relations\\\"\" ws \":\" ws \"[\" ws relations ws \"]\" ws \"}\"\n"
    "entities ::= entity (\",\" ws entity)* | \"\"\n"
    "entity ::= \"{\" ws \"\\\"text\\\"\" ws \":\" ws string ws \",\" ws "
    "\"\\\"type\\\"\" ws \":\" ws string ws \",\" ws "
    "\"\\\"score\\\"\" ws \":\" ws number ws \"}\"\n"
    "relations ::= relation (\",\" ws relation)* | \"\"\n"
    "relation ::= \"{\" ws \"\\\"head\\\"\" ws \":\" ws string ws \",\" ws "
    "\"\\\"rel\\\"\" ws \":\" ws string ws \",\" ws "
    "\"\\\"tail\\\"\" ws \":\" ws string ws \",\" ws "
    "\"\\\"score\\\"\" ws \":\" ws number ws \"}\"\n" GBNF_COMMON_RULES;

/* ──────────────────────────────────────────────────────────────────
 * Few-Shot Examples
 *
 * Two static demonstrations appended to every system prompt.
 * NOT from the test corpus. NOT templated with user labels.
 * Covers: person, organization, location, date, monetary_value,
 *         ceo_of, announced, located_in, acquired, acquired_for.
 * ────────────────────────────────────────────────────────────── */

/* clang-format off */

#define FEW_SHOT_NER                                                                                                   \
    "\n\nExample 1:\n"                                                                                                 \
    "Text: \"Tesla CEO Elon Musk announced the opening of a new Gigafactory in Berlin, Germany in 2021.\"\n"           \
    "Output: {\"entities\":["                                                                                          \
    "{\"text\":\"Elon Musk\",\"type\":\"person\",\"score\":0.99},"                                                     \
    "{\"text\":\"Tesla\",\"type\":\"organization\",\"score\":0.99},"                                                   \
    "{\"text\":\"Berlin\",\"type\":\"location\",\"score\":0.95},"                                                      \
    "{\"text\":\"Germany\",\"type\":\"location\",\"score\":0.99},"                                                     \
    "{\"text\":\"2021\",\"type\":\"date\",\"score\":0.99}]}\n"                                                         \
    "\nExample 2:\n"                                                                                                   \
    "Text: \"Microsoft acquired LinkedIn for $26.2 billion in 2016.\"\n"                                               \
    "Output: {\"entities\":["                                                                                          \
    "{\"text\":\"Microsoft\",\"type\":\"organization\",\"score\":0.99},"                                               \
    "{\"text\":\"LinkedIn\",\"type\":\"organization\",\"score\":0.99},"                                                \
    "{\"text\":\"$26.2 billion\",\"type\":\"monetary_value\",\"score\":0.99},"                                         \
    "{\"text\":\"2016\",\"type\":\"date\",\"score\":0.99}]}"

#define FEW_SHOT_RE                                                                                                    \
    "\n\nExample 1:\n"                                                                                                 \
    "Entities: [{\"text\":\"Elon Musk\",\"type\":\"person\"},"                                                         \
    "{\"text\":\"Tesla\",\"type\":\"organization\"},"                                                                  \
    "{\"text\":\"Berlin\",\"type\":\"location\"},"                                                                     \
    "{\"text\":\"Germany\",\"type\":\"location\"},"                                                                    \
    "{\"text\":\"2021\",\"type\":\"date\"}]\n"                                                                         \
    "Text: \"Tesla CEO Elon Musk announced the opening of a new Gigafactory in Berlin, Germany in 2021.\"\n"           \
    "Output: {\"relations\":["                                                                                         \
    "{\"head\":\"Elon Musk\",\"rel\":\"ceo_of\",\"tail\":\"Tesla\",\"score\":0.99},"                                   \
    "{\"head\":\"Elon Musk\",\"rel\":\"announced\",\"tail\":\"Gigafactory\",\"score\":0.90},"                          \
    "{\"head\":\"Gigafactory\",\"rel\":\"located_in\",\"tail\":\"Berlin\",\"score\":0.95}]}\n"                         \
    "\nExample 2:\n"                                                                                                   \
    "Entities: [{\"text\":\"Microsoft\",\"type\":\"organization\"},"                                                   \
    "{\"text\":\"LinkedIn\",\"type\":\"organization\"},"                                                               \
    "{\"text\":\"$26.2 billion\",\"type\":\"monetary_value\"},"                                                        \
    "{\"text\":\"2016\",\"type\":\"date\"}]\n"                                                                         \
    "Text: \"Microsoft acquired LinkedIn for $26.2 billion in 2016.\"\n"                                               \
    "Output: {\"relations\":["                                                                                         \
    "{\"head\":\"Microsoft\",\"rel\":\"acquired\",\"tail\":\"LinkedIn\",\"score\":0.99},"                              \
    "{\"head\":\"Microsoft\",\"rel\":\"acquired_for\",\"tail\":\"$26.2 billion\",\"score\":0.95}]}"

#define FEW_SHOT_NER_RE                                                                                                \
    "\n\nExample 1:\n"                                                                                                 \
    "Text: \"Tesla CEO Elon Musk announced the opening of a new Gigafactory in Berlin, Germany in 2021.\"\n"           \
    "Output: {\"entities\":["                                                                                          \
    "{\"text\":\"Elon Musk\",\"type\":\"person\",\"score\":0.99},"                                                     \
    "{\"text\":\"Tesla\",\"type\":\"organization\",\"score\":0.99},"                                                   \
    "{\"text\":\"Berlin\",\"type\":\"location\",\"score\":0.95},"                                                      \
    "{\"text\":\"Germany\",\"type\":\"location\",\"score\":0.99},"                                                     \
    "{\"text\":\"2021\",\"type\":\"date\",\"score\":0.99}],"                                                           \
    "\"relations\":["                                                                                                  \
    "{\"head\":\"Elon Musk\",\"rel\":\"ceo_of\",\"tail\":\"Tesla\",\"score\":0.99},"                                   \
    "{\"head\":\"Elon Musk\",\"rel\":\"announced\",\"tail\":\"Gigafactory\",\"score\":0.90},"                          \
    "{\"head\":\"Gigafactory\",\"rel\":\"located_in\",\"tail\":\"Berlin\",\"score\":0.95}]}\n"                         \
    "\nExample 2:\n"                                                                                                   \
    "Text: \"Microsoft acquired LinkedIn for $26.2 billion in 2016.\"\n"                                               \
    "Output: {\"entities\":["                                                                                          \
    "{\"text\":\"Microsoft\",\"type\":\"organization\",\"score\":0.99},"                                               \
    "{\"text\":\"LinkedIn\",\"type\":\"organization\",\"score\":0.99},"                                                \
    "{\"text\":\"$26.2 billion\",\"type\":\"monetary_value\",\"score\":0.99},"                                         \
    "{\"text\":\"2016\",\"type\":\"date\",\"score\":0.99}],"                                                           \
    "\"relations\":["                                                                                                  \
    "{\"head\":\"Microsoft\",\"rel\":\"acquired\",\"tail\":\"LinkedIn\",\"score\":0.99},"                              \
    "{\"head\":\"Microsoft\",\"rel\":\"acquired_for\",\"tail\":\"$26.2 billion\",\"score\":0.95}]}"

/* clang-format on */

/* ──────────────────────────────────────────────────────────────────
 * System Prompts — Supervised (labels provided)
 * ────────────────────────────────────────────────────────────── */

static const char *SYS_NER_SUP =
    "You are a precise named entity recognition system. "
    "Extract entities of the specified types from the text. "
    "For each entity, assign a confidence score between 0.0 and 1.0 reflecting "
    "how certain you are that the span is correctly identified as that entity type. "
    "Use the full range: 1.0 = definite, 0.7-0.9 = high confidence, "
    "0.4-0.6 = moderate, below 0.4 = uncertain. "
    "Respond ONLY with a JSON object in this format: "
    "{\"entities\":[{\"text\":\"entity text\",\"type\":\"entity_type\",\"score\":0.85},...]} "
    FEW_SHOT_NER;

static const char *SYS_RE_SUP =
    "You are a precise relation extraction system. "
    "Given text and a list of entities, extract relations between them. "
    "Only emit relations between the provided entities. "
    "For each relation, assign a confidence score between 0.0 and 1.0 reflecting "
    "how certain you are that this relation is explicitly supported by the text. "
    "Use the full range: 1.0 = explicitly stated, 0.7-0.9 = strongly implied, "
    "0.4-0.6 = inferred, below 0.4 = speculative. "
    "Respond ONLY with a JSON object in this format: "
    "{\"relations\":[{\"head\":\"entity\",\"rel\":\"relation_type\",\"tail\":\"entity\",\"score\":0.75},...]} "
    FEW_SHOT_RE;

static const char *SYS_NER_RE_SUP =
    "You are a precise knowledge extraction system. "
    "Extract named entities and relations between them from the text. "
    "For entities: identify spans matching the specified types with a confidence score [0.0-1.0]. "
    "For relations: identify how entities are connected with a confidence score [0.0-1.0]. "
    "Entity scores: 1.0 = definite, 0.7-0.9 = high, 0.4-0.6 = moderate, below 0.4 = uncertain. "
    "Relation scores: 1.0 = explicitly stated, 0.7-0.9 = strongly implied, "
    "0.4-0.6 = inferred, below 0.4 = speculative. "
    "Respond ONLY with a JSON object in this format: "
    "{\"entities\":[{\"text\":\"entity text\",\"type\":\"entity_type\",\"score\":0.85},...], "
    "\"relations\":[{\"head\":\"entity\",\"rel\":\"relation_type\",\"tail\":\"entity\",\"score\":0.75},...]} "
    FEW_SHOT_NER_RE;

/* ──────────────────────────────────────────────────────────────────
 * System Prompts — Unsupervised (open extraction, no labels)
 * ────────────────────────────────────────────────────────────── */

static const char *SYS_NER_UNSUP =
    "You are a precise named entity recognition system. "
    "Extract ALL notable entities from the text — people, organizations, locations, "
    "dates, products, events, and any other significant named entities. "
    "Choose the most specific type for each entity. "
    "For each entity, assign a confidence score between 0.0 and 1.0 reflecting "
    "how certain you are that the span is correctly identified as that entity type. "
    "Use the full range: 1.0 = definite, 0.7-0.9 = high confidence, "
    "0.4-0.6 = moderate, below 0.4 = uncertain. "
    "Respond ONLY with a JSON object in this format: "
    "{\"entities\":[{\"text\":\"entity text\",\"type\":\"entity_type\",\"score\":0.85},...]} "
    FEW_SHOT_NER;

static const char *SYS_RE_UNSUP =
    "You are a precise relation extraction system. "
    "Given the text, identify ALL notable entities and extract relations between them. "
    "Discover the relationship types from the text itself. "
    "For each relation, assign a confidence score between 0.0 and 1.0 reflecting "
    "how certain you are that this relation is explicitly supported by the text. "
    "Use the full range: 1.0 = explicitly stated, 0.7-0.9 = strongly implied, "
    "0.4-0.6 = inferred, below 0.4 = speculative. "
    "Respond ONLY with a JSON object in this format: "
    "{\"relations\":[{\"head\":\"entity\",\"rel\":\"relation_type\",\"tail\":\"entity\",\"score\":0.75},...]} "
    FEW_SHOT_RE;

static const char *SYS_NER_RE_UNSUP =
    "You are a precise knowledge extraction system. "
    "Extract ALL notable named entities and relations between them from the text. "
    "Discover entity types and relationship types from the text itself. "
    "For entities: identify all significant spans with a confidence score [0.0-1.0]. "
    "For relations: identify how entities are connected with a confidence score [0.0-1.0]. "
    "Entity scores: 1.0 = definite, 0.7-0.9 = high, 0.4-0.6 = moderate, below 0.4 = uncertain. "
    "Relation scores: 1.0 = explicitly stated, 0.7-0.9 = strongly implied, "
    "0.4-0.6 = inferred, below 0.4 = speculative. "
    "Respond ONLY with a JSON object in this format: "
    "{\"entities\":[{\"text\":\"entity text\",\"type\":\"entity_type\",\"score\":0.85},...], "
    "\"relations\":[{\"head\":\"entity\",\"rel\":\"relation_type\",\"tail\":\"entity\",\"score\":0.75},...]} "
    FEW_SHOT_NER_RE;

#endif /* LLAMA_CONSTANTS_H */
