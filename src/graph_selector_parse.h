/*
 * graph_selector_parse.h — dbt-inspired selector expression parser
 *
 * Parses selector strings like "+node", "node+3", "@node", "A B" (union),
 * "A,B" (intersection), "not A" (complement) into an AST.
 *
 * Grammar (EBNF):
 *   expression  = term { SPACE term }                (union)
 *   term        = "not" atom                          (complement)
 *               | atom { "," atom }                   (intersection)
 *   atom        = ["@"] depth_spec                    (@ = build closure)
 *   depth_spec  = [INT "+"] identifier ["+" [INT]]    (depth-limited traversal)
 *   identifier  = [a-zA-Z_][a-zA-Z0-9_.-]*
 */
#ifndef GRAPH_SELECTOR_PARSE_H
#define GRAPH_SELECTOR_PARSE_H

/* AST node types */
typedef enum {
    SEL_NODE,        /* literal node name */
    SEL_ANCESTORS,   /* +node with optional depth */
    SEL_DESCENDANTS, /* node+ with optional depth */
    SEL_BOTH,        /* +node+ or N+node+M */
    SEL_CLOSURE,     /* @node (transitive build closure) */
    SEL_UNION,       /* A B — two children */
    SEL_INTERSECT,   /* A,B — two children */
    SEL_COMPLEMENT,  /* not A — one child */
} SelectorType;

typedef struct SelectorNode {
    SelectorType type;
    char *value;                /* node name (owned, NULL for set-op nodes) */
    int depth_up;               /* ancestor depth limit, -1 = unlimited */
    int depth_down;             /* descendant depth limit, -1 = unlimited */
    struct SelectorNode *left;  /* first child (or only child for complement) */
    struct SelectorNode *right; /* second child (for union/intersect) */
} SelectorNode;

/*
 * Parse a selector expression string into an AST.
 *
 * Returns the root SelectorNode on success, or NULL on error.
 * On error, *pzErrMsg is set to a malloc'd error string (caller frees).
 */
SelectorNode *selector_parse(const char *expr, char **pzErrMsg);

/* Free an AST node and all its children. */
void selector_free(SelectorNode *node);

#endif /* GRAPH_SELECTOR_PARSE_H */
