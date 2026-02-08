import logging
import pickle
from pathlib import Path
from typing import Optional

from rank_bm25 import BM25Okapi

from rag_system.retrieval.dense_retriever import RetrievalResult

logger = logging.getLogger(__name__)


class SparseRetriever:
    def __init__(self):
        self._bm25: Optional[BM25Okapi] = None
        self._chunk_ids: list[str] = []
        self._chunk_data: list[dict] = []
        self._corpus_tokens: list[list[str]] = []

    @property
    def is_indexed(self) -> bool:
        return self._bm25 is not None

    def index_chunks(self, chunks: list) -> int:
        self._chunk_ids = []
        self._chunk_data = []
        self._corpus_tokens = []

        for chunk in chunks:
            if isinstance(chunk, dict):
                text = chunk.get("text", "")
                chunk_id = chunk.get("chunk_id", "")
                data = {
                    "arxiv_id": chunk.get("arxiv_id", ""),
                    "paper_title": chunk.get("paper_title", ""),
                    "section_name": chunk.get("section_name", ""),
                    "text": text,
                }
            else:
                text = getattr(chunk, "text", "")
                chunk_id = getattr(chunk, "chunk_id", "")
                data = {
                    "arxiv_id": getattr(chunk, "arxiv_id", ""),
                    "paper_title": getattr(chunk, "paper_title", ""),
                    "section_name": getattr(chunk, "section_name", ""),
                    "text": text,
                }

            title = data.get("paper_title", "")
            section = data.get("section_name", "")
            enriched = f"{title} {section} {text}"

            tokens = self._tokenize(enriched)
            self._corpus_tokens.append(tokens)
            self._chunk_ids.append(chunk_id)
            self._chunk_data.append(data)

        self._bm25 = BM25Okapi(self._corpus_tokens)
        logger.info(f"BM25 index built with {len(self._chunk_ids)} chunks")
        return len(self._chunk_ids)

    def search(
        self,
        query: str,
        top_k: int = 20,
        exclude_arxiv_ids: Optional[list[str]] = None,
    ) -> list[RetrievalResult]:
        if not self.is_indexed:
            logger.warning("BM25 index not built yet")
            return []

        query_tokens = self._tokenize(query)
        scores = self._bm25.get_scores(query_tokens)

        scored_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )

        results: list[RetrievalResult] = []
        for idx in scored_indices:
            if len(results) >= top_k:
                break

            data = self._chunk_data[idx]
            arxiv_id = data.get("arxiv_id", "")

            if exclude_arxiv_ids and arxiv_id in exclude_arxiv_ids:
                continue

            if scores[idx] <= 0:
                continue

            results.append(
                RetrievalResult(
                    chunk_id=self._chunk_ids[idx],
                    score=float(scores[idx]),
                    text=data.get("text", ""),
                    arxiv_id=arxiv_id,
                    paper_title=data.get("paper_title", ""),
                    section_name=data.get("section_name", ""),
                    metadata=data,
                )
            )

        if results:
            max_score = max(r.score for r in results)
            if max_score > 0:
                for r in results:
                    r.score = r.score / max_score

        return results

    def _tokenize(self, text: str) -> list[str]:
        text = text.lower()
        tokens = text.split()
        return [t for t in tokens if len(t) > 1]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "chunk_ids": self._chunk_ids,
            "chunk_data": self._chunk_data,
            "corpus_tokens": self._corpus_tokens,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        logger.info(f"Saved BM25 index to {path}")

    def load(self, path: str | Path) -> None:
        path = Path(path)
        with open(path, "rb") as f:
            data = pickle.load(f)
        self._chunk_ids = data["chunk_ids"]
        self._chunk_data = data["chunk_data"]
        self._corpus_tokens = data["corpus_tokens"]
        self._bm25 = BM25Okapi(self._corpus_tokens)
        logger.info(f"Loaded BM25 index: {len(self._chunk_ids)} chunks")
