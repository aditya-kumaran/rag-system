import json
import logging
from pathlib import Path
from typing import Optional

from rag_system.evaluation.metrics import (
    EvaluationResult,
    LatencyTracker,
    RAGASMetrics,
    compute_answer_relevance,
    compute_context_precision,
    compute_faithfulness,
    compute_retrieval_metrics,
)

logger = logging.getLogger(__name__)

TEST_QUESTIONS = [
    {"query": "What is Dense Passage Retrieval (DPR)?", "type": "factual", "relevant_papers": ["2004.04906"]},
    {"query": "How does REALM pre-training work?", "type": "factual", "relevant_papers": ["2002.08909"]},
    {"query": "What is Retrieval-Augmented Generation?", "type": "factual", "relevant_papers": ["2005.11401"]},
    {"query": "Explain the FiD (Fusion-in-Decoder) approach", "type": "factual", "relevant_papers": ["2007.01282"]},
    {"query": "What is ColBERT and how does it work?", "type": "factual", "relevant_papers": ["2004.12832"]},
    {"query": "How does RETRO use retrieval for language modeling?", "type": "factual", "relevant_papers": ["2112.04426"]},
    {"query": "What is the Atlas model?", "type": "factual", "relevant_papers": ["2208.03299"]},
    {"query": "Explain Hypothetical Document Embeddings (HyDE)", "type": "factual", "relevant_papers": ["2212.10496"]},
    {"query": "What is DRAGON for dual encoder training?", "type": "factual", "relevant_papers": ["2302.07452"]},
    {"query": "How does Self-RAG work?", "type": "factual", "relevant_papers": ["2310.11511"]},
    {"query": "What is CRAG (Corrective RAG)?", "type": "factual", "relevant_papers": ["2401.15884"]},
    {"query": "Explain the concept of query expansion in neural retrieval", "type": "factual", "relevant_papers": []},
    {"query": "What is contrastive learning for dense retrieval?", "type": "factual", "relevant_papers": []},
    {"query": "How do knowledge graphs augment retrieval?", "type": "factual", "relevant_papers": []},
    {"query": "What is the role of cross-encoders in reranking?", "type": "factual", "relevant_papers": []},
    {"query": "Compare DPR and BM25 for passage retrieval", "type": "comparative", "relevant_papers": ["2004.04906"]},
    {"query": "How does ColBERT compare to dense passage retrieval?", "type": "comparative", "relevant_papers": ["2004.12832", "2004.04906"]},
    {"query": "Compare RAG and FiD approaches to knowledge-intensive NLP", "type": "comparative", "relevant_papers": ["2005.11401", "2007.01282"]},
    {"query": "Dense retrieval vs sparse retrieval: which is better?", "type": "comparative", "relevant_papers": []},
    {"query": "Compare REALM and RETRO pre-training approaches", "type": "comparative", "relevant_papers": ["2002.08909", "2112.04426"]},
    {"query": "How do bi-encoder and cross-encoder models differ?", "type": "comparative", "relevant_papers": []},
    {"query": "Compare Self-RAG and standard RAG approaches", "type": "comparative", "relevant_papers": ["2310.11511", "2005.11401"]},
    {"query": "Query expansion vs query rewriting for retrieval", "type": "comparative", "relevant_papers": []},
    {"query": "Lexical vs semantic matching in document retrieval", "type": "comparative", "relevant_papers": []},
    {"query": "Compare chunking strategies for RAG systems", "type": "comparative", "relevant_papers": []},
    {"query": "What are the recent advances in RAG systems (2024)?", "type": "survey", "relevant_papers": []},
    {"query": "Survey of dense retrieval methods since 2020", "type": "survey", "relevant_papers": []},
    {"query": "Overview of multi-hop question answering approaches", "type": "survey", "relevant_papers": []},
    {"query": "State of the art in open-domain question answering", "type": "survey", "relevant_papers": []},
    {"query": "Recent trends in retrieval-augmented language models", "type": "survey", "relevant_papers": []},
    {"query": "Evolution of passage retrieval from BM25 to neural methods", "type": "survey", "relevant_papers": []},
    {"query": "Survey of hallucination reduction techniques in LLMs", "type": "survey", "relevant_papers": []},
    {"query": "Overview of embedding models for semantic search", "type": "survey", "relevant_papers": []},
    {"query": "Recent work on multi-modal retrieval augmented generation", "type": "survey", "relevant_papers": []},
    {"query": "Survey of evaluation metrics for RAG systems", "type": "survey", "relevant_papers": []},
    {"query": "Papers by Vladimir Karpukhin on retrieval", "type": "author", "relevant_papers": ["2004.04906"]},
    {"query": "What has Patrick Lewis published on RAG?", "type": "author", "relevant_papers": ["2005.11401"]},
    {"query": "Papers by Danqi Chen on dense retrieval", "type": "author", "relevant_papers": []},
    {"query": "Research by Sebastian Riedel on knowledge-intensive NLP", "type": "author", "relevant_papers": []},
    {"query": "Work by Omar Khattab on ColBERT", "type": "author", "relevant_papers": ["2004.12832"]},
    {"query": "How have methods citing DPR improved upon it?", "type": "graph", "relevant_papers": ["2004.04906"]},
    {"query": "What papers build on the RAG framework?", "type": "graph", "relevant_papers": ["2005.11401"]},
    {"query": "Papers that extend REALM's approach", "type": "graph", "relevant_papers": ["2002.08909"]},
    {"query": "What is the citation network around ColBERT?", "type": "graph", "relevant_papers": ["2004.12832"]},
    {"query": "Find papers related to both DPR and RAG", "type": "graph", "relevant_papers": ["2004.04906", "2005.11401"]},
    {"query": "How has the original Transformer paper influenced retrieval?", "type": "temporal", "relevant_papers": []},
    {"query": "Timeline of dense retrieval development 2020-2024", "type": "temporal", "relevant_papers": []},
    {"query": "What were the key breakthroughs in RAG in 2023?", "type": "temporal", "relevant_papers": []},
    {"query": "Evolution of embedding models from Word2Vec to modern transformers", "type": "temporal", "relevant_papers": []},
    {"query": "How has retrieval augmented generation evolved since the original paper?", "type": "temporal", "relevant_papers": ["2005.11401"]},
    {"query": "What is knowledge distillation for retrieval models?", "type": "factual", "relevant_papers": []},
    {"query": "Explain the concept of negative mining in dense retrieval training", "type": "factual", "relevant_papers": []},
    {"query": "What is late interaction in neural retrieval?", "type": "factual", "relevant_papers": ["2004.12832"]},
    {"query": "How do you handle long documents in RAG systems?", "type": "factual", "relevant_papers": []},
    {"query": "What is the role of the reader model in extractive QA?", "type": "factual", "relevant_papers": []},
    {"query": "How does iterative retrieval improve RAG performance?", "type": "factual", "relevant_papers": []},
    {"query": "What are the challenges of deploying RAG systems in production?", "type": "factual", "relevant_papers": []},
    {"query": "How do you evaluate retrieval quality in RAG?", "type": "factual", "relevant_papers": []},
    {"query": "What is the impact of chunk size on RAG performance?", "type": "factual", "relevant_papers": []},
    {"query": "How do instruction-tuned models improve RAG?", "type": "factual", "relevant_papers": []},
    {"query": "Compare FAISS and ChromaDB for vector storage", "type": "comparative", "relevant_papers": []},
    {"query": "Dense retrieval vs hybrid retrieval effectiveness", "type": "comparative", "relevant_papers": []},
    {"query": "Compare sentence-transformers and OpenAI embeddings for retrieval", "type": "comparative", "relevant_papers": []},
    {"query": "Abstractive vs extractive question answering with retrieval", "type": "comparative", "relevant_papers": []},
    {"query": "Compare fine-tuning vs RAG for knowledge-intensive tasks", "type": "comparative", "relevant_papers": []},
    {"query": "What improvements has SPLADE brought to sparse retrieval?", "type": "factual", "relevant_papers": []},
    {"query": "How does document-level vs passage-level retrieval compare?", "type": "comparative", "relevant_papers": []},
    {"query": "What is the BEIR benchmark for retrieval evaluation?", "type": "factual", "relevant_papers": []},
    {"query": "Explain multi-vector retrieval approaches", "type": "factual", "relevant_papers": []},
    {"query": "What are the most cited papers in retrieval augmented generation?", "type": "graph", "relevant_papers": []},
    {"query": "How do citation patterns reflect the evolution of RAG research?", "type": "graph", "relevant_papers": []},
    {"query": "Which research communities focus on different aspects of RAG?", "type": "graph", "relevant_papers": []},
    {"query": "What papers bridge the gap between retrieval and generation?", "type": "graph", "relevant_papers": []},
    {"query": "Find the most influential papers in dense retrieval", "type": "graph", "relevant_papers": []},
    {"query": "How has the field of neural retrieval grown since 2020?", "type": "temporal", "relevant_papers": []},
    {"query": "What were the key papers of 2022 in RAG?", "type": "temporal", "relevant_papers": []},
    {"query": "When did hybrid retrieval become popular?", "type": "temporal", "relevant_papers": []},
    {"query": "What is the trend in model sizes for retrieval?", "type": "temporal", "relevant_papers": []},
    {"query": "How has evaluation methodology evolved for RAG systems?", "type": "temporal", "relevant_papers": []},
    {"query": "What is adaptive retrieval in RAG systems?", "type": "factual", "relevant_papers": []},
    {"query": "How do you handle multi-lingual retrieval in RAG?", "type": "factual", "relevant_papers": []},
    {"query": "What is the role of reinforcement learning in retrieval?", "type": "factual", "relevant_papers": []},
    {"query": "Explain learned sparse retrieval models", "type": "factual", "relevant_papers": []},
    {"query": "What is the re2g approach to retrieval augmented generation?", "type": "factual", "relevant_papers": []},
    {"query": "How do you handle conflicting information from multiple retrieved passages?", "type": "factual", "relevant_papers": []},
    {"query": "What is retrieval-augmented code generation?", "type": "factual", "relevant_papers": []},
    {"query": "Compare end-to-end vs pipeline RAG approaches", "type": "comparative", "relevant_papers": []},
    {"query": "How do different pooling strategies affect retrieval quality?", "type": "factual", "relevant_papers": []},
    {"query": "What is the relationship between pre-training and retrieval performance?", "type": "factual", "relevant_papers": []},
    {"query": "Survey of attention mechanisms used in retrieval models", "type": "survey", "relevant_papers": []},
    {"query": "Overview of data augmentation techniques for training retrievers", "type": "survey", "relevant_papers": []},
    {"query": "Recent advances in zero-shot retrieval", "type": "survey", "relevant_papers": []},
    {"query": "State of the art in conversational question answering with retrieval", "type": "survey", "relevant_papers": []},
    {"query": "Survey of efficient retrieval methods for large-scale systems", "type": "survey", "relevant_papers": []},
    {"query": "How does in-context learning interact with retrieval augmentation?", "type": "factual", "relevant_papers": []},
    {"query": "What are the privacy implications of retrieval augmented generation?", "type": "factual", "relevant_papers": []},
    {"query": "Compare different negative sampling strategies for retriever training", "type": "comparative", "relevant_papers": []},
    {"query": "What is the effect of retrieval on reducing hallucinations?", "type": "factual", "relevant_papers": []},
    {"query": "How do graph neural networks enhance retrieval systems?", "type": "factual", "relevant_papers": []},
    {"query": "What is the optimal number of retrieved passages for RAG?", "type": "factual", "relevant_papers": []},
    {"query": "Compare BM25, TF-IDF, and learned sparse methods", "type": "comparative", "relevant_papers": []},
    {"query": "How do you build a production RAG system?", "type": "factual", "relevant_papers": []},
    {"query": "What is the role of metadata filtering in retrieval?", "type": "factual", "relevant_papers": []},
    {"query": "Explain retrieval-augmented generation for structured data", "type": "factual", "relevant_papers": []},
]


class Evaluator:
    def __init__(self, hybrid_retriever=None, reranker=None, llm_client=None):
        self.hybrid_retriever = hybrid_retriever
        self.reranker = reranker
        self.llm_client = llm_client

    def evaluate_retrieval(
        self,
        test_questions: Optional[list[dict]] = None,
        methods: Optional[list[str]] = None,
    ) -> dict[str, list[EvaluationResult]]:
        if test_questions is None:
            test_questions = [q for q in TEST_QUESTIONS if q.get("relevant_papers")]

        if methods is None:
            methods = ["dense", "sparse", "hybrid", "hybrid_graph"]

        results: dict[str, list[EvaluationResult]] = {m: [] for m in methods}

        for question in test_questions:
            query = question["query"]
            query_type = question.get("type", "general")
            relevant = question.get("relevant_papers", [])

            if not relevant:
                continue

            for method in methods:
                tracker = LatencyTracker()
                tracker.start("retrieval")

                retrieved = self._retrieve(query, method)
                tracker.stop("retrieval")

                retrieved_ids = [r.arxiv_id for r in retrieved if r.arxiv_id]

                retrieval_metrics = compute_retrieval_metrics(retrieved_ids, relevant)

                eval_result = EvaluationResult(
                    query=query,
                    query_type=query_type,
                    retrieval_metrics=retrieval_metrics,
                    latency=tracker.get_metrics(),
                    retrieved_papers=retrieved_ids[:20],
                    method=method,
                )
                results[method].append(eval_result)

        return results

    def evaluate_generation(
        self,
        test_questions: Optional[list[dict]] = None,
        num_questions: int = 20,
    ) -> list[EvaluationResult]:
        if test_questions is None:
            test_questions = TEST_QUESTIONS[:num_questions]

        results: list[EvaluationResult] = []

        for question in test_questions:
            query = question["query"]
            query_type = question.get("type", "general")

            tracker = LatencyTracker()

            tracker.start("retrieval")
            retrieved = self._retrieve(query, "hybrid_graph")
            tracker.stop("retrieval")

            if self.reranker:
                tracker.start("reranking")
                retrieved = self.reranker.rerank(query, retrieved, final_k=10)
                tracker.stop("reranking")

            answer = ""
            if self.llm_client:
                tracker.start("generation")
                answer = self.llm_client.generate(
                    query=query,
                    retrieved_results=retrieved,
                    query_type=query_type,
                )
                tracker.stop("generation")

            context = " ".join([r.text for r in retrieved[:10]])
            ragas = RAGASMetrics(
                faithfulness=compute_faithfulness(answer, context),
                answer_relevance=compute_answer_relevance(answer, query),
                context_precision=compute_context_precision(
                    [r.text for r in retrieved[:10]], query
                ),
                context_recall=0.0,
            )

            eval_result = EvaluationResult(
                query=query,
                query_type=query_type,
                ragas_metrics=ragas,
                latency=tracker.get_metrics(),
                answer=answer,
                retrieved_papers=[r.arxiv_id for r in retrieved if r.arxiv_id][:20],
                method="hybrid_graph",
            )
            results.append(eval_result)

        return results

    def run_ablation_study(
        self,
        test_questions: Optional[list[dict]] = None,
    ) -> dict:
        if test_questions is None:
            test_questions = [q for q in TEST_QUESTIONS if q.get("relevant_papers")]

        methods = ["dense", "sparse", "hybrid", "hybrid_graph"]
        all_results = self.evaluate_retrieval(test_questions, methods)

        summary: dict[str, dict] = {}
        for method, results in all_results.items():
            if not results:
                continue

            metrics_list = [r.retrieval_metrics for r in results if r.retrieval_metrics]
            if not metrics_list:
                continue

            summary[method] = {
                "num_queries": len(metrics_list),
                "avg_precision_at_1": sum(m.precision_at_1 for m in metrics_list) / len(metrics_list),
                "avg_precision_at_5": sum(m.precision_at_5 for m in metrics_list) / len(metrics_list),
                "avg_precision_at_10": sum(m.precision_at_10 for m in metrics_list) / len(metrics_list),
                "avg_recall_at_10": sum(m.recall_at_10 for m in metrics_list) / len(metrics_list),
                "avg_mrr": sum(m.mrr for m in metrics_list) / len(metrics_list),
                "avg_ndcg_at_10": sum(m.ndcg_at_10 for m in metrics_list) / len(metrics_list),
                "avg_latency_ms": sum(r.latency.total_ms for r in results if r.latency) / len(results),
            }

        return summary

    def _retrieve(self, query: str, method: str) -> list:
        if not self.hybrid_retriever:
            return []

        if method == "dense":
            return self.hybrid_retriever.search_dense_only(query, top_k=20)
        elif method == "sparse":
            return self.hybrid_retriever.search_sparse_only(query, top_k=20)
        elif method == "hybrid":
            return self.hybrid_retriever.search_hybrid_no_graph(query, top_k=20)
        elif method == "hybrid_graph":
            return self.hybrid_retriever.search(query, top_k=20)
        return []

    def generate_report(
        self,
        ablation_results: dict,
        generation_results: Optional[list[EvaluationResult]] = None,
        output_path: Optional[str | Path] = None,
    ) -> str:
        report_lines = ["# RAG System Evaluation Report\n"]

        report_lines.append("## Retrieval Ablation Study\n")
        report_lines.append("| Method | P@1 | P@5 | P@10 | R@10 | MRR | nDCG@10 | Latency (ms) |")
        report_lines.append("|--------|-----|-----|------|------|-----|---------|-------------|")

        for method, metrics in ablation_results.items():
            report_lines.append(
                f"| {method} | {metrics['avg_precision_at_1']:.3f} | "
                f"{metrics['avg_precision_at_5']:.3f} | {metrics['avg_precision_at_10']:.3f} | "
                f"{metrics['avg_recall_at_10']:.3f} | {metrics['avg_mrr']:.3f} | "
                f"{metrics['avg_ndcg_at_10']:.3f} | {metrics['avg_latency_ms']:.1f} |"
            )

        if generation_results:
            report_lines.append("\n## Generation Quality (RAGAS Metrics)\n")
            ragas_list = [r.ragas_metrics for r in generation_results if r.ragas_metrics]
            if ragas_list:
                avg_faith = sum(m.faithfulness for m in ragas_list) / len(ragas_list)
                avg_relevance = sum(m.answer_relevance for m in ragas_list) / len(ragas_list)
                avg_ctx_prec = sum(m.context_precision for m in ragas_list) / len(ragas_list)

                report_lines.append(f"- **Faithfulness:** {avg_faith:.3f}")
                report_lines.append(f"- **Answer Relevance:** {avg_relevance:.3f}")
                report_lines.append(f"- **Context Precision:** {avg_ctx_prec:.3f}")

            latencies = [r.latency for r in generation_results if r.latency]
            if latencies:
                avg_total = sum(lat.total_ms for lat in latencies) / len(latencies)
                report_lines.append(f"\n- **Average Total Latency:** {avg_total:.1f} ms")

        report = "\n".join(report_lines)

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                f.write(report)
            logger.info(f"Saved evaluation report to {output_path}")

            json_path = output_path.with_suffix(".json")
            with open(json_path, "w") as f:
                json.dump(ablation_results, f, indent=2, default=str)

        return report
