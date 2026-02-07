import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "all-MiniLM-L6-v2"
SPECTER_MODEL = "allenai/specter2_base"


class EmbeddingGenerator:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        batch_size: int = 32,
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = None
        self._device = device

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device=self._device)
            logger.info(f"Loaded embedding model: {self.model_name}")
        return self._model

    @property
    def dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    def embed_texts(self, texts: list[str], show_progress: bool = True) -> np.ndarray:
        if not texts:
            return np.array([])

        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,
        )
        return np.array(embeddings)

    def embed_query(self, query: str) -> np.ndarray:
        embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
        )
        return np.array(embedding[0])

    def embed_chunks(
        self,
        chunks: list,
        output_path: Optional[str | Path] = None,
    ) -> tuple[list[str], np.ndarray]:
        texts = []
        chunk_ids = []

        for chunk in chunks:
            if isinstance(chunk, dict):
                text = chunk.get("text", "")
                chunk_id = chunk.get("chunk_id", "")
                title = chunk.get("paper_title", "")
                section = chunk.get("section_name", "")
            else:
                text = getattr(chunk, "text", "")
                chunk_id = getattr(chunk, "chunk_id", "")
                title = getattr(chunk, "paper_title", "")
                section = getattr(chunk, "section_name", "")

            enriched = f"Title: {title}\nSection: {section}\n{text}"
            texts.append(enriched)
            chunk_ids.append(chunk_id)

        logger.info(f"Generating embeddings for {len(texts)} chunks...")
        embeddings = self.embed_texts(texts)

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(str(output_path), embeddings)
            ids_path = output_path.with_suffix(".ids.json")
            with open(ids_path, "w") as f:
                json.dump(chunk_ids, f)
            logger.info(f"Saved {len(chunk_ids)} embeddings to {output_path}")

        return chunk_ids, embeddings

    def embed_papers_for_graph(
        self,
        papers_metadata: dict,
    ) -> dict[str, np.ndarray]:
        paper_embeddings: dict[str, np.ndarray] = {}
        arxiv_ids = []
        texts = []

        for arxiv_id, meta in papers_metadata.items():
            if isinstance(meta, dict):
                title = meta.get("title", "")
                abstract = meta.get("abstract", "")
            else:
                title = getattr(meta, "title", "")
                abstract = getattr(meta, "abstract", "")

            combined = f"{title}. {abstract}"
            arxiv_ids.append(arxiv_id)
            texts.append(combined)

        if texts:
            embeddings = self.embed_texts(texts)
            for aid, emb in zip(arxiv_ids, embeddings):
                paper_embeddings[aid] = emb

        logger.info(f"Generated paper-level embeddings for {len(paper_embeddings)} papers")
        return paper_embeddings
