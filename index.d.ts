/**
 * sqlite-muninn — HNSW vector search + graph traversal + Node2Vec for SQLite
 */

/** Package version */
export declare const version: string;

/** Get the absolute path to the muninn loadable extension binary */
export declare function getLoadablePath(): string;

/** A SQLite database connection that supports loading extensions */
interface Db {
  loadExtension(file: string, entrypoint?: string | undefined): void;
}

/** Load the muninn extension into a SQLite database connection */
export declare function load(db: Db): void;
