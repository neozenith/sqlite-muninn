/*
 * llama_label_groups.c — muninn_label_groups TVF
 *
 * Reads a membership table, groups members by group column, constructs
 * prompts, and calls the LLM to generate concise labels for each group.
 *
 * All prompt construction and LLM calling happens in C — no Python
 * orchestration needed. The calling code just does:
 *
 *   INSERT INTO output_table(group_id, label, member_count)
 *   SELECT group_id, label, member_count FROM muninn_label_groups
 *   WHERE model = 'Qwen3.5-4B'
 *     AND membership_table = 'my_view'
 *     AND group_col = 'canonical'
 *     AND member_col = 'name'
 *     AND min_group_size = 3;
 */

#include "llama_label_groups.h"
#include "llama_common.h"
#include "llama_chat.h"
#include "id_validate.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

SQLITE_EXTENSION_INIT3

/* ── Result row ──────────────────────────────────────────────────── */

typedef struct {
    char *group_id;
    char *label;
    int member_count;
} LabelRow;

typedef struct {
    LabelRow *rows;
    int count;
    int capacity;
} LabelResults;

static void lr_init(LabelResults *r) {
    r->count = 0;
    r->capacity = 64;
    r->rows = (LabelRow *)calloc((size_t)r->capacity, sizeof(LabelRow));
}

static void lr_destroy(LabelResults *r) {
    for (int i = 0; i < r->count; i++) {
        free(r->rows[i].group_id);
        free(r->rows[i].label);
    }
    free(r->rows);
    r->rows = NULL;
    r->count = 0;
}

static void lr_add(LabelResults *r, const char *group_id, const char *label, int member_count) {
    if (r->count >= r->capacity) {
        r->capacity *= 2;
        r->rows = (LabelRow *)realloc(r->rows, (size_t)r->capacity * sizeof(LabelRow));
    }
    LabelRow *row = &r->rows[r->count++];
    row->group_id = strdup(group_id);
    row->label = strdup(label);
    row->member_count = member_count;
}

/* ── Column enum ─────────────────────────────────────────────────── */

enum {
    LG_COL_GROUP_ID = 0,
    LG_COL_LABEL,
    LG_COL_MEMBER_COUNT,
    /* Hidden params */
    LG_COL_MODEL,
    LG_COL_MEMBERSHIP_TABLE,
    LG_COL_GROUP_COL,
    LG_COL_MEMBER_COL,
    LG_COL_MIN_GROUP_SIZE,
    LG_COL_MAX_MEMBERS,
    LG_COL_SYSTEM_PROMPT,
};

typedef struct {
    sqlite3_vtab base;
    sqlite3 *db;
} LGVtab;

typedef struct {
    sqlite3_vtab_cursor base;
    LabelResults results;
    int current;
    int eof;
} LGCursor;

/* ── Clean label output ──────────────────────────────────────────── */

static char *clean_label(const char *raw) {
    /* Strip think blocks */
    const char *clean = strip_think_block(raw);

    /* Copy and trim whitespace */
    while (*clean == ' ' || *clean == '\n' || *clean == '\r' || *clean == '\t')
        clean++;
    int len = (int)strlen(clean);
    while (len > 0 && (clean[len - 1] == ' ' || clean[len - 1] == '\n' || clean[len - 1] == '\r'))
        len--;

    /* Strip surrounding quotes */
    if (len >= 2 && clean[0] == clean[len - 1] && (clean[0] == '"' || clean[0] == '\'')) {
        clean++;
        len -= 2;
    }

    char *result = (char *)malloc((size_t)len + 1);
    memcpy(result, clean, (size_t)len);
    result[len] = '\0';
    return result;
}

/* ── VTab methods ────────────────────────────────────────────────── */

static int lg_connect(sqlite3 *db, void *pAux, int argc, const char *const *argv, sqlite3_vtab **ppVtab, char **pzErr) {
    (void)pAux;
    (void)argc;
    (void)argv;
    (void)pzErr;
    int rc = sqlite3_declare_vtab(db, "CREATE TABLE x("
                                      "  group_id TEXT, label TEXT, member_count INTEGER,"
                                      "  model TEXT HIDDEN,"
                                      "  membership_table TEXT HIDDEN,"
                                      "  group_col TEXT HIDDEN,"
                                      "  member_col TEXT HIDDEN,"
                                      "  min_group_size INTEGER HIDDEN,"
                                      "  max_members_in_prompt INTEGER HIDDEN,"
                                      "  system_prompt TEXT HIDDEN"
                                      ")");
    if (rc != SQLITE_OK)
        return rc;

    LGVtab *vtab = (LGVtab *)sqlite3_malloc(sizeof(LGVtab));
    if (!vtab)
        return SQLITE_NOMEM;
    memset(vtab, 0, sizeof(LGVtab));
    vtab->db = db;
    *ppVtab = &vtab->base;
    return SQLITE_OK;
}

static int lg_disconnect(sqlite3_vtab *pVTab) {
    sqlite3_free(pVTab);
    return SQLITE_OK;
}

static int lg_best_index(sqlite3_vtab *pVTab, sqlite3_index_info *pIdxInfo) {
    (void)pVTab;
    int idx_num = 0;
    int argv_idx = 1;

    for (int i = 0; i < pIdxInfo->nConstraint; i++) {
        if (!pIdxInfo->aConstraint[i].usable)
            continue;
        if (pIdxInfo->aConstraint[i].op != SQLITE_INDEX_CONSTRAINT_EQ)
            continue;

        int col = pIdxInfo->aConstraint[i].iColumn;
        if (col >= LG_COL_MODEL && col <= LG_COL_SYSTEM_PROMPT) {
            pIdxInfo->aConstraintUsage[i].argvIndex = argv_idx++;
            pIdxInfo->aConstraintUsage[i].omit = 1;
            idx_num |= (1 << (col - LG_COL_MODEL));
        }
    }

    pIdxInfo->idxNum = idx_num;
    pIdxInfo->estimatedCost = 10000.0;
    return SQLITE_OK;
}

static int lg_open(sqlite3_vtab *pVTab, sqlite3_vtab_cursor **ppCursor) {
    (void)pVTab;
    LGCursor *cur = (LGCursor *)calloc(1, sizeof(LGCursor));
    if (!cur)
        return SQLITE_NOMEM;
    cur->eof = 1;
    *ppCursor = &cur->base;
    return SQLITE_OK;
}

static int lg_close(sqlite3_vtab_cursor *pCursor) {
    LGCursor *cur = (LGCursor *)pCursor;
    lr_destroy(&cur->results);
    free(cur);
    return SQLITE_OK;
}

static int lg_filter(sqlite3_vtab_cursor *pCursor, int idxNum, const char *idxStr, int argc, sqlite3_value **argv) {
    (void)idxStr;
    LGCursor *cur = (LGCursor *)pCursor;
    LGVtab *vtab = (LGVtab *)pCursor->pVtab;

    lr_destroy(&cur->results);
    memset(&cur->results, 0, sizeof(LabelResults));

    /* Parse hidden parameters */
    const char *model_name = NULL;
    const char *membership_table = NULL;
    const char *group_col = NULL;
    const char *member_col = NULL;
    int min_group_size = 3;
    int max_members = 10;
    const char *system_prompt = "Output ONLY a concise label (3-8 words). No explanation.";

    int pos = 0;
    for (int bit = 0; bit < 7 && pos < argc; bit++) {
        if (!(idxNum & (1 << bit)))
            continue;
        int col = bit + LG_COL_MODEL;
        switch (col) {
        case LG_COL_MODEL:
            model_name = (const char *)sqlite3_value_text(argv[pos]);
            break;
        case LG_COL_MEMBERSHIP_TABLE:
            membership_table = (const char *)sqlite3_value_text(argv[pos]);
            break;
        case LG_COL_GROUP_COL:
            group_col = (const char *)sqlite3_value_text(argv[pos]);
            break;
        case LG_COL_MEMBER_COL:
            member_col = (const char *)sqlite3_value_text(argv[pos]);
            break;
        case LG_COL_MIN_GROUP_SIZE:
            min_group_size = sqlite3_value_int(argv[pos]);
            break;
        case LG_COL_MAX_MEMBERS:
            max_members = sqlite3_value_int(argv[pos]);
            break;
        case LG_COL_SYSTEM_PROMPT:
            system_prompt = (const char *)sqlite3_value_text(argv[pos]);
            break;
        }
        pos++;
    }

    if (!model_name || !membership_table || !group_col || !member_col) {
        vtab->base.zErrMsg =
            sqlite3_mprintf("muninn_label_groups: model, membership_table, group_col, and member_col are required");
        cur->eof = 1;
        return SQLITE_ERROR;
    }

    /* Validate identifiers to prevent SQL injection */
    if (id_validate(membership_table) != 0 || id_validate(group_col) != 0 || id_validate(member_col) != 0) {
        vtab->base.zErrMsg = sqlite3_mprintf("muninn_label_groups: invalid identifier in table/column names");
        cur->eof = 1;
        return SQLITE_ERROR;
    }

    /* Find the chat model */
    MuninnModelEntry *me = muninn_registry_find_type(model_name, MUNINN_MODEL_CHAT);
    if (!me) {
        vtab->base.zErrMsg = sqlite3_mprintf("muninn_label_groups: model '%s' not loaded", model_name);
        cur->eof = 1;
        return SQLITE_ERROR;
    }

    /* Query: SELECT group_col, member_col FROM membership_table ORDER BY group_col, member_col */
    char sql[512];
    snprintf(sql, sizeof(sql), "SELECT [%s], [%s] FROM [%s] ORDER BY [%s], [%s]", group_col, member_col,
             membership_table, group_col, member_col);

    sqlite3_stmt *stmt;
    int rc = sqlite3_prepare_v2(vtab->db, sql, -1, &stmt, NULL);
    if (rc != SQLITE_OK) {
        vtab->base.zErrMsg = sqlite3_mprintf("muninn_label_groups: %s", sqlite3_errmsg(vtab->db));
        cur->eof = 1;
        return SQLITE_ERROR;
    }

    /* Group members */
    typedef struct {
        char *group_id;
        char **members;
        int count;
        int capacity;
    } Group;

    Group *groups = NULL;
    int n_groups = 0, cap_groups = 0;

    char *cur_group = NULL;
    while (sqlite3_step(stmt) == SQLITE_ROW) {
        const char *gid = (const char *)sqlite3_column_text(stmt, 0);
        const char *member = (const char *)sqlite3_column_text(stmt, 1);
        if (!gid || !member)
            continue;

        if (!cur_group || strcmp(cur_group, gid) != 0) {
            /* New group */
            if (n_groups >= cap_groups) {
                cap_groups = cap_groups ? cap_groups * 2 : 64;
                groups = (Group *)realloc(groups, (size_t)cap_groups * sizeof(Group));
            }
            Group *g = &groups[n_groups++];
            g->group_id = strdup(gid);
            g->members = NULL;
            g->count = 0;
            g->capacity = 0;
            free(cur_group);
            cur_group = strdup(gid);
        }

        /* Add member to current group */
        Group *g = &groups[n_groups - 1];
        if (g->count >= g->capacity) {
            g->capacity = g->capacity ? g->capacity * 2 : 8;
            g->members = (char **)realloc(g->members, (size_t)g->capacity * sizeof(char *));
        }
        g->members[g->count++] = strdup(member);
    }
    free(cur_group);
    sqlite3_finalize(stmt);

    /* Generate labels for eligible groups */
    lr_init(&cur->results);

    for (int gi = 0; gi < n_groups; gi++) {
        Group *g = &groups[gi];
        if (g->count < min_group_size)
            continue;

        /* Build prompt: list top N members */
        int prompt_size = 512 + g->count * 64;
        char *prompt = (char *)malloc((size_t)prompt_size);
        int ppos = 0;

        ppos +=
            snprintf(prompt + ppos, (size_t)(prompt_size - ppos), "Group '%s' (%d members):\n", g->group_id, g->count);

        int show = g->count < max_members ? g->count : max_members;
        for (int mi = 0; mi < show; mi++) {
            ppos += snprintf(prompt + ppos, (size_t)(prompt_size - ppos), "- %s\n", g->members[mi]);
        }
        if (g->count > max_members) {
            ppos +=
                snprintf(prompt + ppos, (size_t)(prompt_size - ppos), "- ...and %d others\n", g->count - max_members);
        }
        ppos += snprintf(prompt + ppos, (size_t)(prompt_size - ppos),
                         "\nGenerate a concise label (3-8 words) for this group.");

        /* Format with chat template, skip_think=1 */
        char *formatted = format_chat_messages(me, system_prompt, prompt, 1, NULL);
        free(prompt);

        if (!formatted)
            continue;

        /* Generate */
        char errbuf[256];
        char *output = NULL;
        int output_len = 0;
        rc = chat_generate(me, formatted, NULL, 64, &output, &output_len, errbuf, sizeof(errbuf));
        free(formatted);

        if (rc == 0 && output) {
            char *label = clean_label(output);
            lr_add(&cur->results, g->group_id, label, g->count);
            free(label);
            free(output);
        }
    }

    /* Cleanup groups */
    for (int gi = 0; gi < n_groups; gi++) {
        for (int mi = 0; mi < groups[gi].count; mi++)
            free(groups[gi].members[mi]);
        free(groups[gi].members);
        free(groups[gi].group_id);
    }
    free(groups);

    cur->current = 0;
    cur->eof = (cur->results.count == 0);
    return SQLITE_OK;
}

static int lg_next(sqlite3_vtab_cursor *p) {
    LGCursor *cur = (LGCursor *)p;
    cur->current++;
    cur->eof = (cur->current >= cur->results.count);
    return SQLITE_OK;
}

static int lg_eof(sqlite3_vtab_cursor *p) {
    return ((LGCursor *)p)->eof;
}

static int lg_column(sqlite3_vtab_cursor *p, sqlite3_context *ctx, int col) {
    LGCursor *cur = (LGCursor *)p;
    LabelRow *row = &cur->results.rows[cur->current];
    switch (col) {
    case LG_COL_GROUP_ID:
        sqlite3_result_text(ctx, row->group_id, -1, SQLITE_TRANSIENT);
        break;
    case LG_COL_LABEL:
        sqlite3_result_text(ctx, row->label, -1, SQLITE_TRANSIENT);
        break;
    case LG_COL_MEMBER_COUNT:
        sqlite3_result_int(ctx, row->member_count);
        break;
    default:
        sqlite3_result_null(ctx);
        break;
    }
    return SQLITE_OK;
}

static int lg_rowid(sqlite3_vtab_cursor *p, sqlite3_int64 *pRowid) {
    *pRowid = ((LGCursor *)p)->current;
    return SQLITE_OK;
}

static sqlite3_module llama_label_groups_module = {
    .iVersion = 0,
    .xCreate = NULL,
    .xConnect = lg_connect,
    .xBestIndex = lg_best_index,
    .xDisconnect = lg_disconnect,
    .xDestroy = lg_disconnect,
    .xOpen = lg_open,
    .xClose = lg_close,
    .xFilter = lg_filter,
    .xNext = lg_next,
    .xEof = lg_eof,
    .xColumn = lg_column,
    .xRowid = lg_rowid,
};

/* ── Registration ────────────────────────────────────────────────── */

int llama_label_groups_register_module(sqlite3 *db) {
    return sqlite3_create_module(db, "muninn_label_groups", &llama_label_groups_module, NULL);
}
