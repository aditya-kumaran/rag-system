import logging
from datetime import datetime
from typing import Optional

from rag_system.indexing.embedder import EmbeddingGenerator
from rag_system.retrieval.dense_retriever import DenseRetriever, RetrievalResult
from rag_system.retrieval.graph_retriever import GraphRetriever
from rag_system.retrieval.sparse_retriever import SparseRetriever

logger = logging.getLogger(__name__)

QUERY_TYPES = ["factual", "comparative", "survey", "author", "temporal", "graph", "general"]

WEIGHT_SEMANTIC = 0.5
WEIGHT_PAGERANK = 0.2
WEIGHT_GRAPH_PROXIMITY = 0.2
WEIGHT_RECENCY = 0.1


class HybridRetriever:
    def __init__(
        self,
        dense_retriever: DenseRetriever,
        sparse_retriever: SparseRetriever,
        graph_retriever: Optional[GraphRetriever] = None,
        embedder: Optional[EmbeddingGenerator] = None,
        dense_weight: float = 0.5,
        sparse_weight: float = 0.3,
        graph_weight: float = 0.2,
    ):
        self.dense = dense_retriever
        self.sparse = sparse_retriever
        self.graph = graph_retriever
        self.embedder = embedder
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.graph_weight = graph_weight

    def search(
        self,
        query: str,
        top_k: int = 20,
        candidate_k: int = 100,
        query_type: str = "general",
        exclude_arxiv_ids: Optional[list[str]] = None,
        seed_arxiv_ids: Optional[list[str]] = None,
        use_graph: bool = True,
    ) -> list[RetrievalResult]:
        query_embedding = None
        if self.embedder:
            query_embedding = self.embedder.embed_query(query)

        dense_results = []
        if query_embedding is not None:
            dense_results = self.dense.search(
                query_embedding=query_embedding,
                top_k=candidate_k,
                exclude_arxiv_ids=exclude_arxiv_ids,
            )

        sparse_results = self.sparse.search(
            query=query,
            top_k=candidate_k,
            exclude_arxiv_ids=exclude_arxiv_ids,
        )

        graph_results: list[RetrievalResult] = []
        if use_graph and self.graph and seed_arxiv_ids:
            expansion = self.graph.citation_expansion(seed_arxiv_ids, max_results=candidate_k // 3)
            walk = self.graph.graph_walk(
                seed_arxiv_ids, query_embedding=query_embedding,
                max_results=candidate_k // 3,
            )
            community = self.graph.community_retrieval(seed_arxiv_ids, max_results=candidate_k // 3)
            graph_results = expansion + walk + community

        if not seed_arxiv_ids and dense_results:
            seed_ids = list({r.arxiv_id for r in dense_results[:5] if r.arxiv_id})
            if seed_ids and use_graph and self.graph:
                expansion = self.graph.citation_expansion(seed_ids, max_results=candidate_k // 3)
                walk = self.graph.graph_walk(
                    seed_ids, query_embedding=query_embedding,
                    max_results=candidate_k // 3,
                )
                community = self.graph.community_retrieval(seed_ids, max_results=candidate_k // 3)
                graph_results = expansion + walk + community
                seed_arxiv_ids = seed_ids

        fused = self._reciprocal_rank_fusion(
            dense_results, sparse_results, graph_results
        )

        if self.graph and seed_arxiv_ids:
            fused = self._apply_graph_scoring(fused, seed_arxiv_ids)

        fused = self._apply_recency_boost(fused)

        fused.sort(key=lambda r: r.score, reverse=True)
        return fused[:top_k]

    def _reciprocal_rank_fusion(
        self,
        dense_results: list[RetrievalResult],
        sparse_results: list[RetrievalResult],
        graph_results: list[RetrievalResult],
        k: int = 60,
    ) -> list[RetrievalResult]:
        chunk_scores: dict[str, float] = {}
        chunk_map: dict[str, RetrievalResult] = {}

        for rank, result in enumerate(dense_results):
            rrf_score = self.dense_weight / (k + rank + 1)
            key = result.chunk_id
            chunk_scores[key] = chunk_scores.get(key, 0) + rrf_score
            if key not in chunk_map:
                chunk_map[key] = result

        for rank, result in enumerate(sparse_results):
            rrf_score = self.sparse_weight / (k + rank + 1)
            key = result.chunk_id
            chunk_scores[key] = chunk_scores.get(key, 0) + rrf_score
            if key not in chunk_map:
                chunk_map[key] = result

        for rank, result in enumerate(graph_results):
            rrf_score = self.graph_weight / (k + rank + 1)
            key = result.arxiv_id if result.arxiv_id else result.chunk_id
            chunk_scores[key] = chunk_scores.get(key, 0) + rrf_score
            if key not in chunk_map:
                chunk_map[key] = result

        results: list[RetrievalResult] = []
        for key, score in chunk_scores.items():
            if key in chunk_map:
                result = chunk_map[key]
                result.score = score
                results.append(result)

        return results

    def _apply_graph_scoring(
        self,
        results: list[RetrievalResult],
        seed_arxiv_ids: list[str],
    ) -> list[RetrievalResult]:
        for result in results:
            if not result.arxiv_id:
                continue

            graph_scores = self.graph.get_graph_score(result.arxiv_id, seed_arxiv_ids)
            pagerank = graph_scores["pagerank"]
            proximity = graph_scores["graph_proximity"]

            max_pr = max(self.graph.graph_builder.pagerank_scores.values()) if self.graph.graph_builder.pagerank_scores else 1.0
            normalized_pr = pagerank / max_pr if max_pr > 0 else 0

            combined = (
                WEIGHT_SEMANTIC * result.score
                + WEIGHT_PAGERANK * normalized_pr
                + WEIGHT_GRAPH_PROXIMITY * proximity
            )
            result.score = combined
            result.metadata["pagerank"] = pagerank
            result.metadata["graph_proximity"] = proximity

        return results

    def _apply_recency_boost(self, results: list[RetrievalResult]) -> list[RetrievalResult]:
        current_year = datetime.now().year

        for result in results:
            published = result.metadata.get("published", "")
            if not published:
                continue

            try:
                if isinstance(published, str):
                    year = int(published[:4])
                else:
                    year = current_year
            except (ValueError, IndexError):
                year = current_year

            years_old = max(0, current_year - year)
            recency = max(0, 1.0 - (years_old * 0.15))
            result.score = result.score * (1 - WEIGHT_RECENCY) + WEIGHT_RECENCY * recency

        return results

    def classify_query(self, query: str) -> str:
        query_lower = query.lower()

        if any(w in query_lower for w in ["compare", "versus", "vs", "difference between", "better"]):
            return "comparative"

        if any(w in query_lower for w in ["survey", "overview", "recent advances", "state of the art", "review"]):
            return "survey"

        if any(w in query_lower for w in ["papers by", "author", "wrote", "published by"]):
            return "author"

        if any(w in query_lower for w in ["when", "timeline", "history", "evolution", "trend"]):
            return "temporal"

        if any(w in query_lower for w in ["citing", "cited by", "related papers", "builds on", "extends"]):
            return "graph"

        if any(w in query_lower for w in ["what is", "define", "explain", "how does", "describe"]):
            return "factual"

        return "general"

    def search_dense_only(
        self,
        query: str,
        top_k: int = 20,
    ) -> list[RetrievalResult]:
        if self.embedder:
            query_embedding = self.embedder.embed_query(query)
            return self.dense.search(query_embedding=query_embedding, top_k=top_k)
        return []

    def search_sparse_only(
        self,
        query: str,
        top_k: int = 20,
    ) -> list[RetrievalResult]:
        return self.sparse.search(query=query, top_k=top_k)

    def search_hybrid_no_graph(
        self,
        query: str,
        top_k: int = 20,
    ) -> list[RetrievalResult]:
        return self.search(query=query, top_k=top_k, use_graph=False)
