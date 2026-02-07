import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RetrievalMetrics:
    precision_at_1: float = 0.0
    precision_at_5: float = 0.0
    precision_at_10: float = 0.0
    precision_at_20: float = 0.0
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    recall_at_20: float = 0.0
    mrr: float = 0.0
    ndcg_at_10: float = 0.0
    map_score: float = 0.0


@dataclass
class RAGASMetrics:
    faithfulness: float = 0.0
    answer_relevance: float = 0.0
    context_precision: float = 0.0
    context_recall: float = 0.0


@dataclass
class LatencyMetrics:
    embedding_ms: float = 0.0
    retrieval_ms: float = 0.0
    reranking_ms: float = 0.0
    generation_ms: float = 0.0
    total_ms: float = 0.0


@dataclass
class EvaluationResult:
    query: str
    query_type: str
    retrieval_metrics: Optional[RetrievalMetrics] = None
    ragas_metrics: Optional[RAGASMetrics] = None
    latency: Optional[LatencyMetrics] = None
    answer: str = ""
    retrieved_papers: list[str] = field(default_factory=list)
    method: str = ""


def precision_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    if not relevant or k == 0:
        return 0.0
    retrieved_k = retrieved[:k]
    relevant_set = set(relevant)
    hits = sum(1 for doc in retrieved_k if doc in relevant_set)
    return hits / k


def recall_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    if not relevant:
        return 0.0
    retrieved_k = retrieved[:k]
    relevant_set = set(relevant)
    hits = sum(1 for doc in retrieved_k if doc in relevant_set)
    return hits / len(relevant_set)


def mean_reciprocal_rank(retrieved: list[str], relevant: list[str]) -> float:
    relevant_set = set(relevant)
    for i, doc in enumerate(retrieved):
        if doc in relevant_set:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    if not relevant:
        return 0.0

    relevance_scores = []
    relevant_set = set(relevant)
    for doc in retrieved[:k]:
        relevance_scores.append(1.0 if doc in relevant_set else 0.0)

    dcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(relevance_scores))

    ideal_scores = sorted(relevance_scores, reverse=True)
    idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_scores))

    if idcg == 0:
        return 0.0
    return dcg / idcg


def average_precision(retrieved: list[str], relevant: list[str]) -> float:
    if not relevant:
        return 0.0

    relevant_set = set(relevant)
    hits = 0
    sum_precisions = 0.0

    for i, doc in enumerate(retrieved):
        if doc in relevant_set:
            hits += 1
            sum_precisions += hits / (i + 1)

    return sum_precisions / len(relevant_set)


def compute_retrieval_metrics(
    retrieved_ids: list[str],
    relevant_ids: list[str],
) -> RetrievalMetrics:
    return RetrievalMetrics(
        precision_at_1=precision_at_k(retrieved_ids, relevant_ids, 1),
        precision_at_5=precision_at_k(retrieved_ids, relevant_ids, 5),
        precision_at_10=precision_at_k(retrieved_ids, relevant_ids, 10),
        precision_at_20=precision_at_k(retrieved_ids, relevant_ids, 20),
        recall_at_5=recall_at_k(retrieved_ids, relevant_ids, 5),
        recall_at_10=recall_at_k(retrieved_ids, relevant_ids, 10),
        recall_at_20=recall_at_k(retrieved_ids, relevant_ids, 20),
        mrr=mean_reciprocal_rank(retrieved_ids, relevant_ids),
        ndcg_at_10=ndcg_at_k(retrieved_ids, relevant_ids, 10),
        map_score=average_precision(retrieved_ids, relevant_ids),
    )


def compute_faithfulness(answer: str, context: str) -> float:
    if not answer or not context:
        return 0.0

    answer_sentences = [s.strip() for s in answer.split(".") if len(s.strip()) > 10]
    if not answer_sentences:
        return 0.0

    context_lower = context.lower()
    supported = 0
    for sentence in answer_sentences:
        words = sentence.lower().split()
        key_words = [w for w in words if len(w) > 4]
        if not key_words:
            supported += 1
            continue
        overlap = sum(1 for w in key_words if w in context_lower)
        if overlap / len(key_words) > 0.3:
            supported += 1

    return supported / len(answer_sentences)


def compute_answer_relevance(answer: str, query: str) -> float:
    if not answer or not query:
        return 0.0

    query_words = set(query.lower().split())
    answer_words = set(answer.lower().split())

    key_query_words = {w for w in query_words if len(w) > 3}
    if not key_query_words:
        return 0.5

    overlap = key_query_words & answer_words
    return len(overlap) / len(key_query_words)


def compute_context_precision(retrieved_texts: list[str], query: str) -> float:
    if not retrieved_texts or not query:
        return 0.0

    query_words = set(query.lower().split())
    key_words = {w for w in query_words if len(w) > 3}

    if not key_words:
        return 0.5

    relevant_count = 0
    for text in retrieved_texts:
        text_lower = text.lower()
        overlap = sum(1 for w in key_words if w in text_lower)
        if overlap / len(key_words) > 0.2:
            relevant_count += 1

    return relevant_count / len(retrieved_texts)


class LatencyTracker:
    def __init__(self):
        self.timings: dict[str, float] = {}
        self._start_times: dict[str, float] = {}

    def start(self, stage: str) -> None:
        self._start_times[stage] = time.time()

    def stop(self, stage: str) -> float:
        if stage not in self._start_times:
            return 0.0
        elapsed = (time.time() - self._start_times[stage]) * 1000
        self.timings[stage] = elapsed
        return elapsed

    def get_metrics(self) -> LatencyMetrics:
        return LatencyMetrics(
            embedding_ms=self.timings.get("embedding", 0.0),
            retrieval_ms=self.timings.get("retrieval", 0.0),
            reranking_ms=self.timings.get("reranking", 0.0),
            generation_ms=self.timings.get("generation", 0.0),
            total_ms=sum(self.timings.values()),
        )
