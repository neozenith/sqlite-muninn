/*
 * id_validate.h — SQL identifier validation for injection prevention
 *
 * Graph TVFs accept table/column names as parameters and construct
 * SQL dynamically. This module validates those identifiers to prevent
 * SQL injection attacks.
 */
#ifndef ID_VALIDATE_H
#define ID_VALIDATE_H

/*
 * Validate a SQL identifier (table name, column name).
 * Returns 0 if valid, -1 if invalid.
 *
 * Valid identifiers: non-empty, [a-zA-Z0-9_] only.
 */
int id_validate(const char *identifier);

/*
 * Validate and quote a SQL identifier using SQLite's %w format.
 * Returns a sqlite3_mprintf-allocated string that the caller must free
 * with sqlite3_free(). Returns NULL if identifier is invalid.
 *
 * Example: "my_table" -> "\"my_table\""
 */
char *id_quote(const char *identifier);

#endif /* ID_VALIDATE_H */
