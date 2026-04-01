/*
 * llama_label_groups.h — muninn_label_groups TVF
 *
 * Table-valued function that reads a membership table, groups members,
 * and uses an LLM to generate a concise label for each group.
 *
 * SQL:
 *   SELECT group_id, label, member_count FROM muninn_label_groups
 *   WHERE model = 'Qwen3.5-4B'
 *     AND membership_table = 'my_table'
 *     AND group_col = 'cluster_id'
 *     AND member_col = 'member_name'
 *     AND min_group_size = 3
 *     AND max_members_in_prompt = 10
 *     AND system_prompt = 'Output ONLY a concise label (3-8 words).'
 */
#ifndef LLAMA_LABEL_GROUPS_H
#define LLAMA_LABEL_GROUPS_H

#include "sqlite3ext.h"

int llama_label_groups_register_module(sqlite3 *db);

#endif /* LLAMA_LABEL_GROUPS_H */
