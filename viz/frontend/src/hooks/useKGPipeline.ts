/** TanStack Query hooks for KG Pipeline data. */

import { useQuery, useMutation, useInfiniteQuery } from '@tanstack/react-query'
import * as kgService from '@/lib/services/kg-service'
import * as vssService from '@/lib/services/vss-service'
import * as graphService from '@/lib/services/graph-service'

export const kgKeys = {
  pipeline: ['kg', 'pipeline'] as const,
  stage: (num: number) => ['kg', 'stage', num] as const,
  stageItems: (num: number, page: number, pageSize: number, query: string) =>
    ['kg', 'stage-items', num, page, pageSize, query] as const,
  stageItemsInfinite: (num: number, pageSize: number, query: string) =>
    ['kg', 'stage-items-infinite', num, pageSize, query] as const,
  entitiesGrouped: (pageSize: number, query: string) => ['kg', 'entities-grouped', pageSize, query] as const,
  entitiesByChunk: (pageSize: number, query: string) => ['kg', 'entities-by-chunk', pageSize, query] as const,
}

export function usePipeline() {
  return useQuery({
    queryKey: kgKeys.pipeline,
    queryFn: kgService.fetchPipeline,
    staleTime: 60_000,
  })
}

export function useStageDetail(stageNum: number | null) {
  return useQuery({
    queryKey: kgKeys.stage(stageNum ?? 0),
    queryFn: () => kgService.fetchStage(stageNum!),
    enabled: stageNum != null && stageNum >= 1 && stageNum <= 7,
  })
}

export function useGraphRAGQuery() {
  return useMutation({
    mutationFn: ({ query, k, maxDepth }: { query: string; k?: number; maxDepth?: number }) =>
      kgService.queryGraphRAG(query, k, maxDepth),
  })
}

export function useKGSearch() {
  return useMutation({
    mutationFn: ({ query, k }: { query: string; k?: number }) =>
      kgService.queryKGSearch(query, k),
  })
}

/** Embeddings for KG stages: stage 2 → chunks_vec, stage 7 → node2vec_emb */
const STAGE_EMBEDDING_INDEX: Record<number, string> = {
  2: 'chunks_vec',
  7: 'node2vec_emb',
}

export function useKGEmbeddings(stage: number) {
  const indexName = STAGE_EMBEDDING_INDEX[stage] ?? null
  return useQuery({
    queryKey: ['kg', 'embeddings', stage],
    queryFn: () => vssService.fetchEmbeddings(indexName!, 3),
    enabled: !!indexName,
    staleTime: 5 * 60_000,
  })
}

/** Subgraph for KG stages: stage 4 → relations, stage 6 → edges */
const STAGE_GRAPH_TABLE: Record<number, string> = {
  4: 'relations',
  6: 'edges',
}

export function useKGSubgraph(stage: number) {
  const tableName = STAGE_GRAPH_TABLE[stage] ?? null
  return useQuery({
    queryKey: ['kg', 'subgraph', stage],
    queryFn: () => graphService.fetchSubgraph(tableName!, 200),
    enabled: !!tableName,
    staleTime: 5 * 60_000,
  })
}

/** Paginated stage items (legacy — kept for compatibility) */
export function useStageItems(stage: number | null, page: number, pageSize: number, query: string) {
  return useQuery({
    queryKey: kgKeys.stageItems(stage ?? 0, page, pageSize, query),
    queryFn: () => kgService.fetchStageItems(stage!, page, pageSize, query),
    enabled: stage != null && [1, 3, 4, 5].includes(stage),
    staleTime: 30_000,
  })
}

/** Infinite-scroll stage items */
export function useInfiniteStageItems(stage: number | null, pageSize: number, query: string) {
  return useInfiniteQuery({
    queryKey: kgKeys.stageItemsInfinite(stage ?? 0, pageSize, query),
    queryFn: ({ pageParam }) => kgService.fetchStageItems(stage!, pageParam, pageSize, query),
    initialPageParam: 1,
    getNextPageParam: (lastPage) => {
      const totalPages = Math.ceil(lastPage.total / lastPage.page_size)
      return lastPage.page < totalPages ? lastPage.page + 1 : undefined
    },
    enabled: stage != null && [1, 3, 4, 5].includes(stage),
    staleTime: 30_000,
  })
}

/** Infinite-scroll grouped entities (by name + entity_type) */
export function useInfiniteEntitiesGrouped(pageSize: number, query: string) {
  return useInfiniteQuery({
    queryKey: kgKeys.entitiesGrouped(pageSize, query),
    queryFn: ({ pageParam }) => kgService.fetchEntitiesGrouped(pageParam, pageSize, query),
    initialPageParam: 1,
    getNextPageParam: (lastPage) => {
      const totalPages = Math.ceil(lastPage.total / lastPage.page_size)
      return lastPage.page < totalPages ? lastPage.page + 1 : undefined
    },
    staleTime: 30_000,
  })
}

/** Infinite-scroll entities by chunk */
export function useInfiniteEntitiesByChunk(pageSize: number, query: string) {
  return useInfiniteQuery({
    queryKey: kgKeys.entitiesByChunk(pageSize, query),
    queryFn: ({ pageParam }) => kgService.fetchEntitiesByChunk(pageParam, pageSize, query),
    initialPageParam: 1,
    getNextPageParam: (lastPage) => {
      const totalPages = Math.ceil(lastPage.total / lastPage.page_size)
      return lastPage.page < totalPages ? lastPage.page + 1 : undefined
    },
    staleTime: 30_000,
  })
}
