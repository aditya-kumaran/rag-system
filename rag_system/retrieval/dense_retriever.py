import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    chunk_id: str
    score: float
    text: str = ""
    arxiv_id: str = ""
    paper_title: str = ""
    section_name: str = ""
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class DenseRetriever:
    def __init__(
        self,
        collection_name: str = "paper_chunks",
        persist_dir: str = "data/chroma_db",
    ):
        self.collection_name = collection_name
        self.persist_dir = persist_dir
        self._client = None
        self._collection = None

    @property
    def client(self):
        if self._client is None:
            import chromadb

            self._client = chromadb.PersistentClient(path=self.persist_dir)
        return self._client

    @property
    def collection(self):
        if self._collection is None:
            self._collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def index_chunks(
        self,
        chunks: list,
        embeddings: np.ndarray,
        chunk_ids: Optional[list[str]] = None,
    ) -> int:
        if chunk_ids is None:
            chunk_ids = []
            for chunk in chunks:
                if isinstance(chunk, dict):
                    chunk_ids.append(chunk.get("chunk_id", ""))
                else:
                    chunk_ids.append(getattr(chunk, "chunk_id", ""))

        batch_size = 500
        indexed = 0

        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i : i + batch_size]
            batch_embeddings = embeddings[i : i + batch_size]
            batch_ids = chunk_ids[i : i + batch_size]

            documents = []
            metadatas = []

            for chunk in batch_chunks:
                if isinstance(chunk, dict):
                    documents.append(chunk.get("text", ""))
                    metadatas.append({
                        "arxiv_id": chunk.get("arxiv_id", ""),
                        "paper_title": chunk.get("paper_title", ""),
                        "section_name": chunk.get("section_name", ""),
                        "chunk_type": chunk.get("chunk_type", ""),
                        "position": chunk.get("position", 0),
                    })
                else:
                    documents.append(getattr(chunk, "text", ""))
                    metadatas.append({
                        "arxiv_id": getattr(chunk, "arxiv_id", ""),
                        "paper_title": getattr(chunk, "paper_title", ""),
                        "section_name": getattr(chunk, "section_name", ""),
                        "chunk_type": getattr(chunk, "chunk_type", ""),
                        "position": getattr(chunk, "position", 0),
                    })

            self.collection.upsert(
                ids=batch_ids,
                embeddings=batch_embeddings.tolist(),
                documents=documents,
                metadatas=metadatas,
            )
            indexed += len(batch_ids)
            logger.info(f"Indexed {indexed}/{len(chunks)} chunks")

        logger.info(f"Total indexed: {indexed} chunks in collection '{self.collection_name}'")
        return indexed

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 20,
        filter_arxiv_ids: Optional[list[str]] = None,
        exclude_arxiv_ids: Optional[list[str]] = None,
    ) -> list[RetrievalResult]:
        where_filter = None
        if filter_arxiv_ids:
            where_filter = {"arxiv_id": {"$in": filter_arxiv_ids}}

        query_kwargs = {
            "query_embeddings": [query_embedding.tolist()],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where_filter:
            query_kwargs["where"] = where_filter

        results = self.collection.query(**query_kwargs)

        retrieval_results: list[RetrievalResult] = []

        if not results["ids"] or not results["ids"][0]:
            return retrieval_results

        for i, chunk_id in enumerate(results["ids"][0]):
            meta = results["metadatas"][0][i] if results["metadatas"] else {}
            arxiv_id = meta.get("arxiv_id", "")

            if exclude_arxiv_ids and arxiv_id in exclude_arxiv_ids:
                continue

            distance = results["distances"][0][i] if results["distances"] else 0.0
            score = 1.0 - distance

            retrieval_results.append(
                RetrievalResult(
                    chunk_id=chunk_id,
                    score=score,
                    text=results["documents"][0][i] if results["documents"] else "",
                    arxiv_id=arxiv_id,
                    paper_title=meta.get("paper_title", ""),
                    section_name=meta.get("section_name", ""),
                    metadata=meta,
                )
            )

        return retrieval_results

    def get_collection_count(self) -> int:
        return self.collection.count()
