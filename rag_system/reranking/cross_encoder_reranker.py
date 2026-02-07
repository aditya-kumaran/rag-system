import logging
from typing import Optional

from rag_system.retrieval.dense_retriever import RetrievalResult

logger = logging.getLogger(__name__)

DEFAULT_CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str = DEFAULT_CROSS_ENCODER,
        device: Optional[str] = None,
        batch_size: int = 32,
    ):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name, device=self.device)
            logger.info(f"Loaded cross-encoder: {self.model_name}")
        return self._model

    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int = 20,
    ) -> list[RetrievalResult]:
        if not results:
            return []

        if len(results) <= top_k:
            return results

        pairs = []
        for result in results:
            text = result.text[:512] if result.text else ""
            if result.paper_title:
                text = f"{result.paper_title}. {text}"
            pairs.append([query, text])

        scores = self.model.predict(pairs, batch_size=self.batch_size)

        for i, result in enumerate(results):
            result.metadata["original_score"] = result.score
            result.metadata["cross_encoder_score"] = float(scores[i])
            result.score = float(scores[i])

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]


class RerankerPipeline:
    def __init__(
        self,
        cross_encoder: Optional[CrossEncoderReranker] = None,
        stage1_k: int = 100,
        stage2_k: int = 20,
        stage3_k: int = 10,
    ):
        self.cross_encoder = cross_encoder
        self.stage1_k = stage1_k
        self.stage2_k = stage2_k
        self.stage3_k = stage3_k

    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        final_k: int = 10,
    ) -> list[RetrievalResult]:
        stage1_results = results[: self.stage1_k]

        if self.cross_encoder and len(stage1_results) > self.stage2_k:
            stage2_results = self.cross_encoder.rerank(
                query, stage1_results, top_k=self.stage2_k
            )
        else:
            stage2_results = stage1_results[: self.stage2_k]

        return stage2_results[:final_k]
