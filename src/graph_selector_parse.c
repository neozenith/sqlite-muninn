/*
 * graph_selector_parse.c — Tokenizer + recursive-descent parser for
 * dbt-inspired selector expressions.
 *
 * Produces an AST (SelectorNode tree) consumed by graph_selector_eval.c.
 */
#include "graph_selector_parse.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ═══════════════════════════════════════════════════════════════
 * Token types
 * ═══════════════════════════════════════════════════════════════ */

typedef enum {
    TOK_IDENT, /* identifier or number */
    TOK_PLUS,  /* + */
    TOK_AT,    /* @ */
    TOK_COMMA, /* , */
    TOK_NOT,   /* keyword "not" */
    TOK_EOF,   /* end of input */
} TokenType;

typedef struct {
    TokenType type;
    const char *start; /* pointer into original string */
    int len;
} Token;

/* ═══════════════════════════════════════════════════════════════
 * Tokenizer
 * ═══════════════════════════════════════════════════════════════ */

typedef struct {
    const char *input;
    const char *pos;
    Token current;
    char *errmsg;
} Lexer;

static int is_ident_start(char c) {
    return isalpha((unsigned char)c) || c == '_';
}

static int is_ident_char(char c) {
    return isalnum((unsigned char)c) || c == '_' || c == '.' || c == '-';
}

static int is_digit(char c) {
    return isdigit((unsigned char)c);
}

static void skip_whitespace(Lexer *lex) {
    while (*lex->pos == ' ' || *lex->pos == '\t')
        lex->pos++;
}

/* Advance to next token. Returns 0 on success, -1 on error. */
static int lex_next(Lexer *lex) {
    skip_whitespace(lex);

    if (*lex->pos == '\0') {
        lex->current.type = TOK_EOF;
        lex->current.start = lex->pos;
        lex->current.len = 0;
        return 0;
    }

    char c = *lex->pos;

    if (c == '+') {
        lex->current.type = TOK_PLUS;
        lex->current.start = lex->pos;
        lex->current.len = 1;
        lex->pos++;
        return 0;
    }

    if (c == '@') {
        lex->current.type = TOK_AT;
        lex->current.start = lex->pos;
        lex->current.len = 1;
        lex->pos++;
        return 0;
    }

    if (c == ',') {
        lex->current.type = TOK_COMMA;
        lex->current.start = lex->pos;
        lex->current.len = 1;
        lex->pos++;
        return 0;
    }

    if (is_ident_start(c) || is_digit(c)) {
        const char *start = lex->pos;
        while (is_ident_char(*lex->pos) || is_digit(*lex->pos))
            lex->pos++;

        int len = (int)(lex->pos - start);

        /* Check for "not" keyword (must be followed by space or @ or + or digit) */
        if (len == 3 && strncmp(start, "not", 3) == 0) {
            /* Peek ahead: "not" is a keyword only if followed by whitespace, @, +, digit, or ident */
            skip_whitespace(lex);
            char next = *lex->pos;
            if (next == '\0' || next == ',' || next == ')') {
                /* "not" at end or before comma — treat as identifier */
                lex->current.type = TOK_IDENT;
                lex->current.start = start;
                lex->current.len = len;
                return 0;
            }
            lex->current.type = TOK_NOT;
            lex->current.start = start;
            lex->current.len = 3;
            return 0;
        }

        lex->current.type = TOK_IDENT;
        lex->current.start = start;
        lex->current.len = len;
        return 0;
    }

    /* Unrecognized character */
    lex->errmsg = (char *)malloc(128);
    if (lex->errmsg)
        snprintf(lex->errmsg, 128, "graph_select: unexpected character '%c' at position %d", c,
                 (int)(lex->pos - lex->input));
    return -1;
}

/* Check if current token is an integer (all digits) */
static int tok_is_int(const Token *tok) {
    if (tok->type != TOK_IDENT || tok->len == 0)
        return 0;
    for (int i = 0; i < tok->len; i++) {
        if (!is_digit(tok->start[i]))
            return 0;
    }
    return 1;
}

static int tok_to_int(const Token *tok) {
    int val = 0;
    for (int i = 0; i < tok->len; i++)
        val = val * 10 + (tok->start[i] - '0');
    return val;
}

/* Extract token text as a malloc'd string */
static char *tok_strdup(const Token *tok) {
    char *s = (char *)malloc((size_t)(tok->len + 1));
    if (!s)
        return NULL;
    memcpy(s, tok->start, (size_t)tok->len);
    s[tok->len] = '\0';
    return s;
}

/* ═══════════════════════════════════════════════════════════════
 * AST helpers
 * ═══════════════════════════════════════════════════════════════ */

static SelectorNode *node_alloc(SelectorType type) {
    SelectorNode *n = (SelectorNode *)calloc(1, sizeof(SelectorNode));
    if (!n)
        return NULL;
    n->type = type;
    n->depth_up = -1;
    n->depth_down = -1;
    return n;
}

void selector_free(SelectorNode *node) {
    if (!node)
        return;
    free(node->value);
    selector_free(node->left);
    selector_free(node->right);
    free(node);
}

/* ═══════════════════════════════════════════════════════════════
 * Recursive-descent parser
 *
 * Precedence (lowest to highest):
 *   1. union (space-separated)
 *   2. complement ("not")
 *   3. intersection (comma-separated)
 *   4. atom (node with optional depth/@ prefixes)
 * ═══════════════════════════════════════════════════════════════ */

/* Forward declarations */
static SelectorNode *parse_expression(Lexer *lex);
static SelectorNode *parse_term(Lexer *lex);
static SelectorNode *parse_atom(Lexer *lex);

/*
 * atom = ["@"] depth_spec
 * depth_spec = [INT "+"] identifier ["+" [INT]]
 *
 * Examples:
 *   node           → SEL_NODE
 *   +node          → SEL_ANCESTORS (depth_up = -1)
 *   node+          → SEL_DESCENDANTS (depth_down = -1)
 *   +node+         → SEL_BOTH (both unlimited)
 *   2+node         → SEL_ANCESTORS (depth_up = 2)
 *   node+3         → SEL_DESCENDANTS (depth_down = 3)
 *   2+node+3       → SEL_BOTH (depth_up = 2, depth_down = 3)
 *   @node          → SEL_CLOSURE
 */
static SelectorNode *parse_atom(Lexer *lex) {
    int has_at = 0;
    int has_prefix_plus = 0;
    int depth_up = -1;

    /* Check for @ prefix */
    if (lex->current.type == TOK_AT) {
        has_at = 1;
        if (lex_next(lex) != 0)
            return NULL;
    }

    /* Check for prefix + or INT+ */
    if (lex->current.type == TOK_PLUS) {
        /* Just a + prefix: unlimited ancestors */
        has_prefix_plus = 1;
        depth_up = -1;
        if (lex_next(lex) != 0)
            return NULL;
    } else if (tok_is_int(&lex->current)) {
        /* Could be INT+node or just a numeric node name */
        int val = tok_to_int(&lex->current);
        /* Save position to backtrack if no + follows */
        const char *saved_pos = lex->pos;
        Token saved_tok = lex->current;

        if (lex_next(lex) != 0)
            return NULL;

        if (lex->current.type == TOK_PLUS) {
            /* It's INT+node */
            depth_up = val;
            has_prefix_plus = 1;
            if (lex_next(lex) != 0)
                return NULL;
        } else {
            /* It's a numeric identifier — backtrack */
            lex->pos = saved_pos;
            lex->current = saved_tok;
        }
    }

    /* Expect an identifier (the node name) */
    if (lex->current.type != TOK_IDENT) {
        lex->errmsg = (char *)malloc(128);
        if (lex->errmsg)
            snprintf(lex->errmsg, 128, "graph_select: expected node name at position %d",
                     (int)(lex->current.start - lex->input));
        return NULL;
    }

    char *name = tok_strdup(&lex->current);
    if (!name)
        return NULL;

    if (lex_next(lex) != 0) {
        free(name);
        return NULL;
    }

    /* Check for suffix + or +INT
     *
     * Disambiguation: "node +other" is "node UNION +other", not "node+ other".
     * After consuming the +, if the next token starts a new term (IDENT that
     * isn't a digit, PLUS, AT, NOT), then the + was actually the prefix of
     * the next union term. We must backtrack.
     */
    int has_suffix_plus = 0;
    int depth_down = -1;

    if (lex->current.type == TOK_PLUS) {
        /* Save position to backtrack if this + is really a prefix */
        const char *saved_pos = lex->pos;
        Token saved_tok = lex->current;

        if (lex_next(lex) != 0) {
            free(name);
            return NULL;
        }

        if (tok_is_int(&lex->current)) {
            /* node+N — definitely a suffix with depth limit */
            has_suffix_plus = 1;
            depth_down = tok_to_int(&lex->current);
            if (lex_next(lex) != 0) {
                free(name);
                return NULL;
            }
        } else if (lex->current.type == TOK_EOF || lex->current.type == TOK_COMMA) {
            /* node+ at end or before comma — suffix + (unlimited) */
            has_suffix_plus = 1;
        } else {
            /* + followed by IDENT, PLUS, AT, NOT — this + is prefix of next
             * term, not our suffix. Backtrack. */
            lex->pos = saved_pos;
            lex->current = saved_tok;
        }
    }

    /* Determine node type */
    SelectorNode *node;

    if (has_at) {
        node = node_alloc(SEL_CLOSURE);
        if (!node) {
            free(name);
            return NULL;
        }
        node->value = name;
        /* @ may have prefix/suffix modifiers too, but canonically
         * @node means "descendants + all ancestors of descendants".
         * We allow @+node etc. to be an error for now — just @node. */
        return node;
    }

    if (has_prefix_plus && has_suffix_plus) {
        node = node_alloc(SEL_BOTH);
        if (!node) {
            free(name);
            return NULL;
        }
        node->value = name;
        node->depth_up = depth_up;
        node->depth_down = depth_down;
        return node;
    }

    if (has_prefix_plus) {
        node = node_alloc(SEL_ANCESTORS);
        if (!node) {
            free(name);
            return NULL;
        }
        node->value = name;
        node->depth_up = depth_up;
        return node;
    }

    if (has_suffix_plus) {
        node = node_alloc(SEL_DESCENDANTS);
        if (!node) {
            free(name);
            return NULL;
        }
        node->value = name;
        node->depth_down = depth_down;
        return node;
    }

    /* Bare node */
    node = node_alloc(SEL_NODE);
    if (!node) {
        free(name);
        return NULL;
    }
    node->value = name;
    return node;
}

/*
 * term = "not" atom
 *      | atom { "," atom }     (intersection)
 */
static SelectorNode *parse_term(Lexer *lex) {
    /* Check for "not" */
    if (lex->current.type == TOK_NOT) {
        if (lex_next(lex) != 0)
            return NULL;
        SelectorNode *child = parse_atom(lex);
        if (!child)
            return NULL;

        SelectorNode *node = node_alloc(SEL_COMPLEMENT);
        if (!node) {
            selector_free(child);
            return NULL;
        }
        node->left = child;
        return node;
    }

    SelectorNode *left = parse_atom(lex);
    if (!left)
        return NULL;

    /* Comma-separated intersection */
    while (lex->current.type == TOK_COMMA) {
        if (lex_next(lex) != 0) {
            selector_free(left);
            return NULL;
        }
        SelectorNode *right = parse_atom(lex);
        if (!right) {
            selector_free(left);
            return NULL;
        }

        SelectorNode *intersect = node_alloc(SEL_INTERSECT);
        if (!intersect) {
            selector_free(left);
            selector_free(right);
            return NULL;
        }
        intersect->left = left;
        intersect->right = right;
        left = intersect;
    }

    return left;
}

/*
 * expression = term { SPACE term }     (union)
 *
 * Space-separated terms form a union. We detect "next term" by seeing
 * that after parsing one term, the current token is something that can
 * start a new term (IDENT, PLUS, AT, NOT) and is NOT EOF/comma.
 */
static SelectorNode *parse_expression(Lexer *lex) {
    SelectorNode *left = parse_term(lex);
    if (!left)
        return NULL;

    /* While current token can start a new term... */
    while (lex->current.type == TOK_IDENT || lex->current.type == TOK_PLUS || lex->current.type == TOK_AT ||
           lex->current.type == TOK_NOT) {
        SelectorNode *right = parse_term(lex);
        if (!right) {
            selector_free(left);
            return NULL;
        }

        SelectorNode *u = node_alloc(SEL_UNION);
        if (!u) {
            selector_free(left);
            selector_free(right);
            return NULL;
        }
        u->left = left;
        u->right = right;
        left = u;
    }

    return left;
}

/* ═══════════════════════════════════════════════════════════════
 * Public API
 * ═══════════════════════════════════════════════════════════════ */

SelectorNode *selector_parse(const char *expr, char **pzErrMsg) {
    if (!expr || !*expr) {
        if (pzErrMsg)
            *pzErrMsg = strdup("graph_select: empty selector expression");
        return NULL;
    }

    Lexer lex;
    memset(&lex, 0, sizeof(lex));
    lex.input = expr;
    lex.pos = expr;

    /* Prime the lexer with the first token */
    if (lex_next(&lex) != 0) {
        if (pzErrMsg)
            *pzErrMsg = lex.errmsg;
        else
            free(lex.errmsg);
        return NULL;
    }

    SelectorNode *root = parse_expression(&lex);
    if (!root) {
        if (pzErrMsg)
            *pzErrMsg = lex.errmsg;
        else
            free(lex.errmsg);
        return NULL;
    }

    /* Should be at EOF */
    if (lex.current.type != TOK_EOF) {
        if (pzErrMsg) {
            *pzErrMsg = (char *)malloc(128);
            if (*pzErrMsg)
                snprintf(*pzErrMsg, 128, "graph_select: unexpected token at position %d",
                         (int)(lex.current.start - lex.input));
        }
        selector_free(root);
        return NULL;
    }

    return root;
}
