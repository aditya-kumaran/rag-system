# RAG System - Graph-Augmented Retrieval for ArXiv Papers

Production-grade Retrieval-Augmented Generation system with conversational capabilities and graph-augmented retrieval, built on a knowledge base of 500 ArXiv papers on RAG, retrieval, and LLMs.

## Architecture

```
Query → Query Rewriting (coreference resolution, HyDE)
      → Hybrid Retrieval (Dense + Sparse + Graph)
      → Multi-Stage Reranking (Cross-Encoder)
      → LLM Generation (Groq/OpenAI)
      → Conversational Response with Citations
```

### Retrieval Pipeline

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Dense Retrieval | ChromaDB + sentence-transformers | Semantic similarity search |
| Sparse Retrieval | BM25 (rank-bm25) | Lexical keyword matching |
| Graph Retrieval | NetworkX | Citation-based paper discovery |
| Reranking | cross-encoder/ms-marco-MiniLM-L-6-v2 | Relevance refinement |
| Generation | Groq (Llama 3.3 70B) / GPT-4o-mini | Answer generation with citations |

### Graph-Augmented Retrieval

Three methods combining citation graph structure with semantic search:

1. **Citation Expansion** - Retrieve papers cited by and citing seed papers (1-2 hops)
2. **Graph Walk** - Biased random walk toward semantically relevant nodes
3. **Community Retrieval** - Papers from same citation cluster (Louvain communities)

Score fusion: `0.5 × semantic + 0.2 × PageRank + 0.2 × graph_proximity + 0.1 × recency`

### Conversational Features

- Session management with SQLite persistence
- Coreference resolution ("What are its limitations?" → resolves "its")
- Context compression for long conversations
- Entity tracking (acronyms, paper titles, methods)
- Multi-turn retrieval with paper history awareness

## Project Structure

```
rag_system/
├── data_collection/
│   ├── arxiv_scraper.py          # ArXiv API paper collection (500 papers)
│   ├── pdf_extractor.py          # PyMuPDF text/section extraction
│   └── citation_parser.py        # Reference parsing → citation edges
├── indexing/
│   ├── chunker.py                # Hierarchical chunking (section → paragraph)
│   ├── embedder.py               # Embedding generation (sentence-transformers)
│   └── graph_builder.py          # NetworkX citation graph + PageRank + communities
├── retrieval/
│   ├── dense_retriever.py        # ChromaDB vector search
│   ├── sparse_retriever.py       # BM25 keyword search
│   ├── graph_retriever.py        # Citation expansion, graph walk, community retrieval
│   └── hybrid_retriever.py       # RRF fusion + graph scoring + recency boost
├── reranking/
│   └── cross_encoder_reranker.py # Multi-stage reranking pipeline
├── generation/
│   ├── llm_client.py             # Groq/OpenAI/Together API client
│   └── prompt_templates.py       # Query-type-specific prompts (factual/comparative/survey)
├── conversation/
│   ├── session_manager.py        # SQLite session persistence
│   ├── query_rewriter.py         # Coreference resolution + context injection
│   └── context_compressor.py     # Conversation summarization
├── evaluation/
│   ├── metrics.py                # P@k, MRR, nDCG, RAGAS (faithfulness, relevance)
│   └── evaluator.py              # Ablation studies (dense/sparse/hybrid/graph), 100+ test Qs
├── api/
│   └── main.py                   # FastAPI: /query, /search, /paper, /citations, /graph/stats
├── ui/
│   └── app.py                    # Gradio: chat, search, paper details, citation graph viz
├── pipeline.py                   # Main orchestrator (build index, query, load/save)
└── run_pipeline.py               # CLI entry point
```

## Quick Start

### 1. Install Dependencies

```bash
pip install poetry
poetry install
```

### 2. Configure API Keys

```bash
cp .env.example .env
# Edit .env with your Groq API key (free tier)
```

### 3. Run Full Pipeline

```bash
# Collect papers, extract text, build index
python -m rag_system.run_pipeline all
```

### 4. Launch Services

```bash
# FastAPI server
python -m rag_system.run_pipeline api

# Gradio UI (separate terminal)
python -m rag_system.run_pipeline ui
```

### 5. Docker

```bash
docker-compose up
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/query` | POST | Conversational Q&A with RAG |
| `/search` | POST | Paper search (dense/sparse/hybrid/graph) |
| `/paper/{arxiv_id}` | GET | Paper metadata + graph metrics |
| `/citations/{arxiv_id}` | GET | Citation network for a paper |
| `/graph/stats` | GET | Citation graph statistics |
| `/sessions` | GET | List conversation sessions |
| `/health` | GET | Health check |

## Evaluation

### Test Suite

100+ questions across 6 categories:
- **Factual**: "What is Dense Passage Retrieval?"
- **Comparative**: "Compare DPR and BM25"
- **Survey**: "Recent advances in RAG (2024)?"
- **Author-based**: "Papers by Danqi Chen on retrieval"
- **Graph**: "How do papers citing REALM improve it?"
- **Temporal**: "Timeline of dense retrieval development"

### Ablation Study

Run with:
```bash
python -m rag_system.run_pipeline evaluate
```

Compares: Dense only vs Sparse only vs Hybrid vs Hybrid+Graph

### Metrics

- Retrieval: P@1, P@5, P@10, P@20, Recall@k, MRR, nDCG@10
- Generation: Faithfulness, Answer Relevance, Context Precision (RAGAS-style)
- Latency: Per-stage breakdown (embedding, retrieval, reranking, generation)

## Technology Stack

| Component | Choice | Cost |
|-----------|--------|------|
| PDF Extraction | PyMuPDF | Free |
| Graph DB | NetworkX | Free |
| Vector DB | ChromaDB | Free |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) | Free |
| Sparse Search | rank-bm25 | Free |
| Cross-Encoder | ms-marco-MiniLM-L-6-v2 | Free |
| LLM | Groq free tier (Llama 3.3 70B) | Free |
| Session Store | SQLite | Free |
| API | FastAPI | Free |
| UI | Gradio | Free |

**Total estimated cost: $0/month** (all local/free-tier)

## Configuration

Key parameters in `pipeline.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_papers` | 500 | Number of ArXiv papers to collect |
| `max_chunk_tokens` | 512 | Maximum tokens per chunk |
| `overlap_ratio` | 0.2 | Chunk overlap ratio |
| `embedding_model` | all-MiniLM-L6-v2 | Sentence transformer model |
| `llm_provider` | groq | LLM provider (groq/openai/together) |
| `dense_weight` | 0.5 | Weight for dense retrieval in fusion |
| `sparse_weight` | 0.3 | Weight for sparse retrieval in fusion |
| `graph_weight` | 0.2 | Weight for graph retrieval in fusion |
