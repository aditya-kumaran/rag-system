import json
import logging
from pathlib import Path
from typing import Optional

from rag_system.conversation.context_compressor import ContextCompressor
from rag_system.conversation.query_rewriter import QueryRewriter
from rag_system.conversation.session_manager import SessionManager, SessionState
from rag_system.evaluation.metrics import LatencyTracker
from rag_system.generation.llm_client import LLMClient
from rag_system.indexing.chunker import HierarchicalChunker, chunk_all_papers
from rag_system.indexing.embedder import EmbeddingGenerator
from rag_system.indexing.graph_builder import CitationGraphBuilder
from rag_system.reranking.cross_encoder_reranker import CrossEncoderReranker, RerankerPipeline
from rag_system.retrieval.dense_retriever import DenseRetriever, RetrievalResult
from rag_system.retrieval.graph_retriever import GraphRetriever
from rag_system.retrieval.hybrid_retriever import HybridRetriever
from rag_system.retrieval.sparse_retriever import SparseRetriever

logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = Path("data")


class RAGPipeline:
    def __init__(
        self,
        data_dir: str | Path = DEFAULT_DATA_DIR,
        embedding_model: str = "all-MiniLM-L6-v2",
        llm_provider: str = "groq",
        llm_model: Optional[str] = None,
        use_cross_encoder: bool = True,
    ):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.embedder = EmbeddingGenerator(model_name=embedding_model)
        self.dense_retriever = DenseRetriever(
            persist_dir=str(self.data_dir / "chroma_db")
        )
        self.sparse_retriever = SparseRetriever()
        self.graph_builder = CitationGraphBuilder()
        self.graph_retriever: Optional[GraphRetriever] = None

        self.cross_encoder: Optional[CrossEncoderReranker] = None
        self.reranker: Optional[RerankerPipeline] = None
        if use_cross_encoder:
            self.cross_encoder = CrossEncoderReranker()
            self.reranker = RerankerPipeline(cross_encoder=self.cross_encoder)

        self.llm_client = LLMClient(provider=llm_provider, model=llm_model)
        self.session_manager = SessionManager(
            db_path=str(self.data_dir / "sessions.db")
        )
        self.query_rewriter = QueryRewriter(llm_client=self.llm_client)
        self.context_compressor = ContextCompressor(llm_client=self.llm_client)

        self.hybrid_retriever: Optional[HybridRetriever] = None
        self.papers_metadata: dict = {}
        self._query_cache: dict[str, dict] = {}
        self._cache_max_size = 1000

    def build_index(
        self,
        papers_metadata: dict,
        extracted_papers: dict,
        citation_data: dict,
    ) -> None:
        self.papers_metadata = papers_metadata

        logger.info("Building citation graph...")
        self.graph_builder.build_graph(papers_metadata, citation_data)
        self.graph_builder.compute_pagerank()
        self.graph_builder.detect_communities()
        self.graph_builder.save(str(self.data_dir / "graph"))

        logger.info("Generating paper-level embeddings for graph retrieval...")
        paper_embeddings = self.embedder.embed_papers_for_graph(papers_metadata)

        self.graph_retriever = GraphRetriever(
            graph_builder=self.graph_builder,
            paper_embeddings=paper_embeddings,
        )

        logger.info("Chunking papers...")
        chunker = HierarchicalChunker()
        chunks = chunk_all_papers(extracted_papers, papers_metadata, chunker)
        logger.info(f"Created {len(chunks)} chunks")

        logger.info("Generating chunk embeddings...")
        chunk_ids, embeddings = self.embedder.embed_chunks(
            chunks,
            output_path=self.data_dir / "embeddings" / "chunk_embeddings.npy",
        )

        logger.info("Indexing in ChromaDB...")
        self.dense_retriever.index_chunks(chunks, embeddings, chunk_ids)

        logger.info("Building BM25 index...")
        self.sparse_retriever.index_chunks(chunks)
        self.sparse_retriever.save(str(self.data_dir / "bm25_index.pkl"))

        self.hybrid_retriever = HybridRetriever(
            dense_retriever=self.dense_retriever,
            sparse_retriever=self.sparse_retriever,
            graph_retriever=self.graph_retriever,
            embedder=self.embedder,
        )

        meta_path = self.data_dir / "papers_metadata.json"
        with open(meta_path, "w") as f:
            meta_serializable = {}
            for aid, meta in papers_metadata.items():
                if hasattr(meta, "__dict__"):
                    from dataclasses import asdict

                    meta_serializable[aid] = asdict(meta)
                elif isinstance(meta, dict):
                    meta_serializable[aid] = meta
                else:
                    meta_serializable[aid] = str(meta)
            json.dump(meta_serializable, f, indent=2, default=str)

        logger.info("Index build complete!")

    @classmethod
    def load_from_disk(
        cls,
        data_dir: str | Path = DEFAULT_DATA_DIR,
        embedding_model: str = "all-MiniLM-L6-v2",
        llm_provider: str = "groq",
        use_cross_encoder: bool = True,
    ) -> Optional["RAGPipeline"]:
        data_dir = Path(data_dir)

        if not (data_dir / "chroma_db").exists():
            logger.warning(f"No index found at {data_dir}/chroma_db")
            return None

        pipeline = cls(
            data_dir=data_dir,
            embedding_model=embedding_model,
            llm_provider=llm_provider,
            use_cross_encoder=use_cross_encoder,
        )

        meta_path = data_dir / "papers_metadata.json"
        if meta_path.exists():
            with open(meta_path, "r") as f:
                pipeline.papers_metadata = json.load(f)
            logger.info(f"Loaded metadata for {len(pipeline.papers_metadata)} papers")

        graph_dir = data_dir / "graph"
        if graph_dir.exists():
            pipeline.graph_builder.load(str(graph_dir))

            paper_embeddings = pipeline.embedder.embed_papers_for_graph(
                pipeline.papers_metadata
            )
            pipeline.graph_retriever = GraphRetriever(
                graph_builder=pipeline.graph_builder,
                paper_embeddings=paper_embeddings,
            )

        bm25_path = data_dir / "bm25_index.pkl"
        if bm25_path.exists():
            pipeline.sparse_retriever.load(str(bm25_path))

        pipeline.hybrid_retriever = HybridRetriever(
            dense_retriever=pipeline.dense_retriever,
            sparse_retriever=pipeline.sparse_retriever,
            graph_retriever=pipeline.graph_retriever,
            embedder=pipeline.embedder,
        )

        logger.info("Pipeline loaded from disk")
        return pipeline

    def query(
        self,
        query: str,
        conversation_id: Optional[str] = None,
        top_k: int = 10,
        use_graph: bool = True,
        use_reranking: bool = True,
    ) -> dict:
        cache_key = f"{query}:{conversation_id}:{use_graph}:{use_reranking}"
        if cache_key in self._query_cache:
            return self._query_cache[cache_key]

        tracker = LatencyTracker()

        session = None
        if conversation_id:
            session = self.session_manager.get_session(conversation_id)
        if session is None:
            session = self.session_manager.create_session()

        session.add_message("user", query)

        search_query = query
        if len(session.messages) > 2:
            search_query = self.query_rewriter.rewrite(query, session)

        query_type = "general"
        if self.hybrid_retriever:
            query_type = self.hybrid_retriever.classify_query(search_query)

        tracker.start("retrieval")
        retrieved = self._retrieve(
            search_query,
            session=session,
            top_k=top_k * 5,
            use_graph=use_graph,
        )
        tracker.stop("retrieval")

        if use_reranking and self.reranker and retrieved:
            tracker.start("reranking")
            retrieved = self.reranker.rerank(search_query, retrieved, final_k=top_k)
            tracker.stop("reranking")
        else:
            retrieved = retrieved[:top_k]

        retrieved_ids = list({r.arxiv_id for r in retrieved if r.arxiv_id})
        session.add_papers_discussed(retrieved_ids)

        self._extract_entities(query, retrieved, session)

        tracker.start("generation")
        conversation_context = ""
        if len(session.messages) > 3:
            conversation_context = self.context_compressor.compress(session)

        answer = self.llm_client.generate(
            query=search_query,
            retrieved_results=retrieved,
            query_type=query_type,
            conversation_context=conversation_context,
        )
        tracker.stop("generation")

        session.add_message("assistant", answer, metadata={"arxiv_ids": retrieved_ids})
        self.session_manager.save_session(session)

        result = {
            "answer": answer,
            "conversation_id": session.conversation_id,
            "retrieved": retrieved,
            "query_type": query_type,
            "rewritten_query": search_query,
            "latency": tracker.get_metrics(),
        }

        if len(self._query_cache) < self._cache_max_size:
            self._query_cache[cache_key] = result

        return result

    def search(
        self,
        query: str,
        top_k: int = 20,
        method: str = "hybrid",
    ) -> list[RetrievalResult]:
        if not self.hybrid_retriever:
            return []

        if method == "dense":
            return self.hybrid_retriever.search_dense_only(query, top_k)
        elif method == "sparse":
            return self.hybrid_retriever.search_sparse_only(query, top_k)
        elif method == "hybrid":
            return self.hybrid_retriever.search_hybrid_no_graph(query, top_k)
        elif method == "hybrid_graph":
            return self.hybrid_retriever.search(query, top_k=top_k)
        return []

    def get_paper(self, arxiv_id: str) -> Optional[dict]:
        if arxiv_id not in self.papers_metadata:
            base_id = arxiv_id.split("v")[0] if "v" in arxiv_id else arxiv_id
            if base_id not in self.papers_metadata:
                return None
            arxiv_id = base_id

        meta = self.papers_metadata[arxiv_id]
        if isinstance(meta, dict):
            result = meta.copy()
        else:
            from dataclasses import asdict

            result = asdict(meta)

        result["citation_count"] = self.graph_builder.get_citation_count(arxiv_id)
        result["pagerank_score"] = self.graph_builder.pagerank_scores.get(arxiv_id, 0.0)
        result["community_id"] = self.graph_builder.communities.get(arxiv_id)

        return result

    def get_citations(self, arxiv_id: str) -> Optional[dict]:
        if arxiv_id not in self.graph_builder.graph:
            return None

        cited_by_ids = self.graph_builder.get_papers_citing(arxiv_id)
        cites_ids = self.graph_builder.get_papers_cited_by(arxiv_id)
        community_ids = self.graph_builder.get_community_members(arxiv_id)

        def _paper_info(aid: str) -> dict:
            node = self.graph_builder.graph.nodes.get(aid, {})
            return {
                "arxiv_id": aid,
                "title": node.get("title", "Unknown"),
                "pagerank": self.graph_builder.pagerank_scores.get(aid, 0.0),
            }

        return {
            "arxiv_id": arxiv_id,
            "cited_by": [_paper_info(a) for a in cited_by_ids],
            "cites": [_paper_info(a) for a in cites_ids],
            "community_members": [_paper_info(a) for a in community_ids[:20]],
        }

    def get_graph_stats(self) -> dict:
        return self.graph_builder.get_graph_stats()

    def list_sessions(self) -> list[dict]:
        return self.session_manager.list_sessions()

    def delete_session(self, conversation_id: str) -> None:
        self.session_manager.delete_session(conversation_id)

    def _retrieve(
        self,
        query: str,
        session: Optional[SessionState] = None,
        top_k: int = 50,
        use_graph: bool = True,
    ) -> list[RetrievalResult]:
        if not self.hybrid_retriever:
            return []

        exclude_ids = None
        seed_ids = None

        if session and session.papers_discussed:
            seed_ids = session.papers_discussed[-5:]

        return self.hybrid_retriever.search(
            query=query,
            top_k=top_k,
            exclude_arxiv_ids=exclude_ids,
            seed_arxiv_ids=seed_ids,
            use_graph=use_graph,
        )

    def _extract_entities(
        self,
        query: str,
        results: list[RetrievalResult],
        session: SessionState,
    ) -> None:
        import re

        acronym_pattern = re.compile(r"\b([A-Z]{2,})\b")
        acronyms = acronym_pattern.findall(query)

        for result in results[:5]:
            text = result.text if hasattr(result, "text") else ""
            for acronym in acronyms:
                pattern = re.compile(
                    rf"({acronym})\s*[\(\[]([^)]+)[\)\]]|([^(]+)\s*[\(\[]({acronym})[\)\]]",
                    re.IGNORECASE,
                )
                match = pattern.search(text)
                if match:
                    if match.group(1) and match.group(2):
                        session.add_entity(match.group(1), match.group(2))
                    elif match.group(3) and match.group(4):
                        session.add_entity(match.group(4), match.group(3).strip())

        well_known = {
            "DPR": "Dense Passage Retrieval",
            "RAG": "Retrieval-Augmented Generation",
            "BM25": "BM25 (Best Match 25)",
            "REALM": "Retrieval-Augmented Language Model",
            "FiD": "Fusion-in-Decoder",
            "HyDE": "Hypothetical Document Embeddings",
            "ColBERT": "Contextualized Late Interaction over BERT",
            "RETRO": "Retrieval Enhanced Transformer",
        }
        for acronym in acronyms:
            if acronym in well_known and acronym not in session.entities:
                session.add_entity(acronym, well_known[acronym])
