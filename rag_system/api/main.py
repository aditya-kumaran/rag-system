import logging
import time
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

app = FastAPI(
    title="RAG System API",
    description="Production-grade RAG system with graph-augmented retrieval for ArXiv papers",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_system = None


def get_system():
    global _system
    if _system is None:
        from rag_system.pipeline import RAGPipeline

        _system = RAGPipeline.load_from_disk()
    return _system


class QueryRequest(BaseModel):
    query: str
    conversation_id: Optional[str] = None
    top_k: int = Field(default=10, ge=1, le=50)
    use_graph: bool = True
    use_reranking: bool = True


class QueryResponse(BaseModel):
    answer: str
    conversation_id: str
    papers: list[dict]
    query_type: str
    latency_ms: float


class SearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=20, ge=1, le=100)
    method: str = Field(default="hybrid", pattern="^(dense|sparse|hybrid|hybrid_graph)$")


class SearchResponse(BaseModel):
    results: list[dict]
    total: int
    method: str
    latency_ms: float


class PaperResponse(BaseModel):
    arxiv_id: str
    title: str
    authors: list[str]
    abstract: str
    published: str
    categories: list[str]
    citation_count: int
    pagerank_score: float
    community_id: Optional[int] = None


class CitationResponse(BaseModel):
    arxiv_id: str
    cited_by: list[dict]
    cites: list[dict]
    community_members: list[dict]


@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": time.time()}


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    system = get_system()
    if system is None:
        raise HTTPException(status_code=503, detail="System not initialized")

    start = time.time()

    try:
        result = system.query(
            query=request.query,
            conversation_id=request.conversation_id,
            top_k=request.top_k,
            use_graph=request.use_graph,
            use_reranking=request.use_reranking,
        )

        latency = (time.time() - start) * 1000

        papers = []
        seen = set()
        for r in result.get("retrieved", []):
            aid = r.arxiv_id if hasattr(r, "arxiv_id") else r.get("arxiv_id", "")
            if aid and aid not in seen:
                seen.add(aid)
                papers.append({
                    "arxiv_id": aid,
                    "title": r.paper_title if hasattr(r, "paper_title") else r.get("paper_title", ""),
                    "score": r.score if hasattr(r, "score") else r.get("score", 0),
                    "section": r.section_name if hasattr(r, "section_name") else r.get("section_name", ""),
                })

        return QueryResponse(
            answer=result.get("answer", ""),
            conversation_id=result.get("conversation_id", ""),
            papers=papers,
            query_type=result.get("query_type", "general"),
            latency_ms=latency,
        )

    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    system = get_system()
    if system is None:
        raise HTTPException(status_code=503, detail="System not initialized")

    start = time.time()

    try:
        results = system.search(
            query=request.query,
            top_k=request.top_k,
            method=request.method,
        )

        latency = (time.time() - start) * 1000

        formatted = []
        for r in results:
            formatted.append({
                "chunk_id": r.chunk_id if hasattr(r, "chunk_id") else r.get("chunk_id", ""),
                "arxiv_id": r.arxiv_id if hasattr(r, "arxiv_id") else r.get("arxiv_id", ""),
                "title": r.paper_title if hasattr(r, "paper_title") else r.get("paper_title", ""),
                "section": r.section_name if hasattr(r, "section_name") else r.get("section_name", ""),
                "score": r.score if hasattr(r, "score") else r.get("score", 0),
                "text": (r.text if hasattr(r, "text") else r.get("text", ""))[:500],
            })

        return SearchResponse(
            results=formatted,
            total=len(formatted),
            method=request.method,
            latency_ms=latency,
        )

    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/paper/{arxiv_id}", response_model=PaperResponse)
async def get_paper(arxiv_id: str):
    system = get_system()
    if system is None:
        raise HTTPException(status_code=503, detail="System not initialized")

    try:
        paper = system.get_paper(arxiv_id)
        if paper is None:
            raise HTTPException(status_code=404, detail=f"Paper {arxiv_id} not found")
        return paper

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get paper failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/citations/{arxiv_id}", response_model=CitationResponse)
async def get_citations(arxiv_id: str):
    system = get_system()
    if system is None:
        raise HTTPException(status_code=503, detail="System not initialized")

    try:
        citations = system.get_citations(arxiv_id)
        if citations is None:
            raise HTTPException(status_code=404, detail=f"Paper {arxiv_id} not found")
        return citations

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get citations failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/graph/stats")
async def graph_stats():
    system = get_system()
    if system is None:
        raise HTTPException(status_code=503, detail="System not initialized")

    try:
        return system.get_graph_stats()
    except Exception as e:
        logger.error(f"Graph stats failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sessions")
async def list_sessions():
    system = get_system()
    if system is None:
        raise HTTPException(status_code=503, detail="System not initialized")

    return system.list_sessions()


@app.delete("/session/{conversation_id}")
async def delete_session(conversation_id: str):
    system = get_system()
    if system is None:
        raise HTTPException(status_code=503, detail="System not initialized")

    system.delete_session(conversation_id)
    return {"status": "deleted"}
