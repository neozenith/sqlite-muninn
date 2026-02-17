/** KG Pipeline API service. */

import type {
  GraphRAGResult,
  KGSearchResult,
  PipelineSummary,
  StageDetail,
  StageItemsResponse,
  GroupedEntitiesResponse,
  ChunkEntitiesResponse,
} from '../types'
import { fetchJSON } from './api-client'

export function fetchPipeline(): Promise<PipelineSummary> {
  return fetchJSON<PipelineSummary>('/api/kg/pipeline')
}

export function fetchStage(stageNum: number): Promise<StageDetail> {
  return fetchJSON<StageDetail>(`/api/kg/stage/${stageNum}`)
}

export function queryGraphRAG(query: string, k: number = 10, maxDepth: number = 2): Promise<GraphRAGResult> {
  return fetchJSON<GraphRAGResult>('/api/kg/query', {
    method: 'POST',
    body: JSON.stringify({ query, k, max_depth: maxDepth }),
  })
}

export function queryKGSearch(
  query: string,
  k: number = 10,
): Promise<KGSearchResult> {
  return fetchJSON<KGSearchResult>('/api/kg/query', {
    method: 'POST',
    body: JSON.stringify({ query, k }),
  })
}

export function fetchStageItems(
  stage: number,
  page: number = 1,
  pageSize: number = 20,
  query: string = '',
): Promise<StageItemsResponse> {
  const params = new URLSearchParams({
    page: String(page),
    page_size: String(pageSize),
  })
  if (query) params.set('q', query)
  return fetchJSON<StageItemsResponse>(`/api/kg/stage/${stage}/items?${params}`)
}

export function fetchEntitiesGrouped(
  page: number = 1,
  pageSize: number = 20,
  query: string = '',
): Promise<GroupedEntitiesResponse> {
  const params = new URLSearchParams({
    page: String(page),
    page_size: String(pageSize),
  })
  if (query) params.set('q', query)
  return fetchJSON<GroupedEntitiesResponse>(`/api/kg/stage/3/entities-grouped?${params}`)
}

export function fetchEntitiesByChunk(
  page: number = 1,
  pageSize: number = 20,
  query: string = '',
): Promise<ChunkEntitiesResponse> {
  const params = new URLSearchParams({
    page: String(page),
    page_size: String(pageSize),
  })
  if (query) params.set('q', query)
  return fetchJSON<ChunkEntitiesResponse>(`/api/kg/stage/3/entities-by-chunk?${params}`)
}
