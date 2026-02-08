# Architecture Documentation

## System Overview

This is a production-grade Retrieval-Augmented Generation (RAG) system that answers questions about AI/ML research papers using a combination of semantic search, keyword search, and citation graph analysis. The knowledge base consists of 500 ArXiv papers on RAG, dense retrieval, and LLMs (2020-2025).

## End-to-End Pipeline

```
User Query
    |
    v
[Query Rewriter] -- resolves coreferences, injects conversation context
    |
    v
[Hybrid Retriever] -- combines 3 retrieval methods:
    |-- Dense (ChromaDB, sentence-transformers embeddings)
    |-- Sparse (BM25 keyword matching)
    |-- Graph (citation expansion, graph walk, community retrieval)
    |
    v
[Score Fusion] -- RRF + weighted combination:
    0.5 x semantic_similarity + 0.2 x PageRank + 0.2 x graph_proximity + 0.1 x recency
    |
    v
[Cross-Encoder Reranker] -- ms-marco-MiniLM-L-6-v2 refines top candidates
    |
    v
[LLM Generation] -- Groq API (Llama 3.3 70B) generates cited answer
    |
    v
[Session Manager] -- stores conversation history, tracks papers discussed
    |
    v
Cited Response with ArXiv IDs
```

## Phase 1: Data Collection

### ArXiv Paper Collection (`data_collection/arxiv_scraper.py`)

**Technology:** ArXiv API via the `arxiv` Python library

**What it does:**
- Searches ArXiv for papers matching terms: "retrieval augmented generation", "dense retrieval", "neural information retrieval", "question answering retrieval"
- Filters to papers published between 2020-2025
- Downloads up to 500 paper PDFs
- Stores metadata (title, authors, abstract, ArXiv ID, categories, publication date)

**Output:** `data/papers_metadata.json` + PDF files in `data/pdfs/`

### PDF Text Extraction (`data_collection/pdf_extractor.py`)

**Technology:** PyMuPDF (fitz)

**What it does:**
- Extracts full text from each downloaded PDF
- Detects document sections (Abstract, Introduction, Methods, Results, Discussion, Conclusion, References) using heading patterns and font analysis
- Preserves section boundaries for hierarchical chunking

**Why PyMuPDF:** Faster than pdfplumber, better at detecting section headers via font metadata, handles multi-column layouts common in academic papers.

**Output:** Extracted text per paper with section boundaries in `data/papers/`

### Citation Parsing (`data_collection/citation_parser.py`)

**Technology:** Regex-based reference extraction + fuzzy title matching

**What it does:**
- Extracts the References section from each paper
- Parses individual citations to identify ArXiv IDs and paper titles
- Matches citations against the known paper corpus using fuzzy string matching
- Builds directed edges: Paper A cites Paper B

**Output:** `data/citation_data.json` containing citation edges between papers

## Phase 2: Indexing

### Hierarchical Chunking (`indexing/chunker.py`)

**Technology:** Custom implementation

**What it does:**
- Splits each paper by section (Abstract, Introduction, Methods, etc.)
- Further splits long sections into paragraph-level chunks
- Maintains parent-child relationships (paper -> section -> paragraph)
- Each chunk carries metadata: ArXiv ID, paper title, section name, position, chunk type

**Configuration:**
- Maximum chunk size: 512 tokens
- Overlap ratio: 20% between adjacent chunks within a section
- Generates unique chunk IDs: `{arxiv_id}_{sanitized_section_name}` with deduplication suffixes

### Embedding Generation (`indexing/embedder.py`)

**Technology:** sentence-transformers (`all-MiniLM-L6-v2`)

**What it does:**
- Generates 384-dimensional dense vectors for each text chunk
- Also generates paper-level embeddings (from abstracts) for graph retrieval similarity computations
- Batch processing for efficiency

**Why all-MiniLM-L6-v2:** Fast inference, good general-purpose semantic similarity, small model size (80MB). The system is designed to swap in `allenai/specter2` for better academic paper embeddings if needed.

**Output:** Embeddings stored in ChromaDB and as numpy arrays in `data/embeddings/`

### Citation Graph Construction (`indexing/graph_builder.py`)

**Technology:** NetworkX (directed graph)

**What it does:**
- Builds a directed graph where nodes are papers and edges are citations
- Computes PageRank scores for each paper (measures influence in the citation network)
- Detects communities using the Louvain algorithm (groups of closely-related papers)
- Stores node attributes: title, authors, abstract, ArXiv ID, publication date
- Calculates graph statistics: density, connected components, degree distribution

**Why NetworkX:** Free, no server needed, sufficient for 500-paper graphs, easy serialization. Neo4j would be overkill for this scale and adds deployment complexity.

**Output:** Serialized graph in `data/graph/`

## Phase 3: Retrieval

### Dense Retrieval (`retrieval/dense_retriever.py`)

**Technology:** ChromaDB (persistent vector database)

**What it does:**
- Stores chunk embeddings in a ChromaDB collection with cosine similarity
- Queries the collection with the embedded user query
- Returns top-k chunks ranked by semantic similarity
- Supports filtering by ArXiv ID and excluding already-seen papers

**Why ChromaDB:** Free, local, persistent storage, built-in cosine similarity, easy to use. No server process needed (embedded mode).

### Sparse Retrieval (`retrieval/sparse_retriever.py`)

**Technology:** rank-bm25 (BM25 Okapi)

**What it does:**
- Builds an inverted index over all chunk texts
- Scores chunks by term frequency and inverse document frequency (TF-IDF variant)
- Catches keyword matches that semantic search might miss (exact terms, acronyms, author names)

**Why BM25:** Complements dense retrieval by capturing exact lexical matches. For example, searching "DPR" will match the exact acronym even if the embedding doesn't fully capture it.

**Output:** Serialized BM25 index in `data/bm25_index.pkl`

### Graph Retrieval (`retrieval/graph_retriever.py`)

**Technology:** NetworkX + numpy

**What it does (3 methods):**

1. **Citation Expansion:** Given seed papers from initial retrieval, retrieves papers they cite and papers that cite them (1-2 hops). Surfaces related work not found by text search alone.

2. **Biased Graph Walk:** Starting from seed papers, performs random walks biased toward nodes whose embeddings are semantically similar to the query. Discovers papers connected through citation chains.

3. **Community Retrieval:** Identifies which Louvain community the seed papers belong to, then retrieves other papers from the same community. Finds topically related papers that may not directly cite each other.

**Scoring:** Each graph-retrieved paper gets a score combining:
- PageRank (0.2 weight) - paper influence
- Graph proximity (0.2 weight) - citation distance from seed papers
- Semantic similarity (0.5 weight) - embedding similarity to query
- Recency (0.1 weight) - preference for newer papers

### Hybrid Retrieval (`retrieval/hybrid_retriever.py`)

**Technology:** Custom Reciprocal Rank Fusion (RRF) implementation

**What it does:**
- Runs dense, sparse, and graph retrieval in parallel
- Classifies the query type (factual, comparative, survey, author-based, temporal, graph)
- Merges results using Reciprocal Rank Fusion: `score = sum(1 / (k + rank))` across methods
- Applies the weighted score fusion formula for graph-augmented results
- Deduplicates results by chunk ID

## Phase 4: Reranking

### Cross-Encoder Reranking (`reranking/cross_encoder_reranker.py`)

**Technology:** cross-encoder/ms-marco-MiniLM-L-6-v2 (via sentence-transformers)

**What it does:**
- Takes the top ~50 candidates from hybrid retrieval
- Scores each (query, chunk_text) pair through a cross-encoder that jointly attends to both inputs
- Much more accurate than bi-encoder similarity but slower (hence used only on candidates)
- Returns the top-k results after reranking

**Why cross-encoder:** Bi-encoders (used in dense retrieval) encode query and document independently. Cross-encoders see both together, capturing fine-grained relevance signals. The ms-marco model is trained on passage ranking and works well for academic text.

**Pipeline:** Retrieval (top 50) -> Cross-encoder reranking (top 10) -> Final results

## Phase 5: Generation

### LLM Client (`generation/llm_client.py`)

**Technology:** Groq API (primary), OpenAI API, Together AI (alternatives)

**What it does:**
- Sends the reranked context + query to an LLM for answer generation
- Supports multiple providers: Groq (free tier, Llama 3.3 70B), OpenAI (GPT-4o-mini), Together AI
- Falls back to an extractive response if the LLM API is unavailable
- Includes a HyDE (Hypothetical Document Embeddings) generator for query expansion

**Why Groq:** Free tier provides fast inference on Llama 3.3 70B, which is sufficient for academic QA. The system loads API keys from `.env` via python-dotenv.

### Prompt Templates (`generation/prompt_templates.py`)

**What it does:**
- Provides query-type-specific prompts:
  - **Factual:** Direct answer with paper citations
  - **Comparative:** Structured comparison (architecture, performance, strengths/weaknesses)
  - **Survey:** Chronological overview of developments
- System prompt enforces citation format (ArXiv IDs), recency preference, and uncertainty acknowledgment
- Includes instructions to avoid repeating information in multi-turn conversations

## Phase 6: Conversational Features

### Session Manager (`conversation/session_manager.py`)

**Technology:** SQLite

**What it does:**
- Creates and persists conversation sessions with unique IDs
- Stores full message history (role, content, timestamp, metadata)
- Tracks papers discussed in each session (prevents re-retrieving the same papers)
- Tracks entities mentioned (acronym expansions like DPR -> Dense Passage Retrieval)

**Why SQLite:** Zero-configuration, file-based, sufficient for single-server deployment. Sessions persist across restarts.

### Query Rewriter (`conversation/query_rewriter.py`)

**Technology:** Rule-based + LLM-based rewriting

**What it does:**
- Detects when a query references previous conversation (pronouns, implicit follow-ups)
- **Rule-based:** Replaces pronouns ("it", "this method") with the last mentioned entity. Detects follow-up patterns ("give examples", "elaborate", "how about") and appends the previous topic.
- **LLM-based:** Sends conversation history + current query to the LLM to produce a self-contained rewritten query
- Example: "What are its limitations?" with prior context about DPR -> "What are the limitations of Dense Passage Retrieval?"

### Context Compressor (`conversation/context_compressor.py`)

**Technology:** Extractive summarization + LLM compression

**What it does:**
- For short conversations (<=5 messages): passes full history to the LLM
- For longer conversations: compresses older messages into a summary while keeping recent messages verbatim
- Extracts key topics, papers mentioned, and key exchanges
- Prevents the LLM context window from being overwhelmed by long conversations

## Phase 7: Evaluation

### Metrics (`evaluation/metrics.py`)

**Technology:** Custom implementations + RAGAS-style metrics

**Retrieval metrics:**
- Precision@k (k=1, 5, 10, 20): fraction of retrieved results that are relevant
- Recall@k: fraction of relevant results that are retrieved
- MRR (Mean Reciprocal Rank): position of first relevant result
- nDCG@10: ranking quality metric

**Generation metrics (RAGAS-style):**
- Faithfulness: whether the answer is grounded in retrieved context
- Answer Relevance: whether the answer addresses the question
- Context Precision: whether retrieved context is relevant to the question

**Latency tracking:** Per-stage timing (embedding, retrieval, reranking, generation)

### Evaluator (`evaluation/evaluator.py`)

**What it does:**
- Maintains a test set of 100+ questions across 6 categories (factual, comparative, survey, author-based, graph, temporal)
- Runs ablation studies comparing: Dense only, Sparse only, Hybrid (no graph), Hybrid + Graph
- Generates a markdown evaluation report with results tables
- Measures impact of each retrieval method and reranking stage

## Phase 8: Production & Deployment

### FastAPI Application (`api/main.py`)

**Technology:** FastAPI + Uvicorn

**Endpoints:**
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/query` | POST | Conversational Q&A with full RAG pipeline |
| `/search` | POST | Paper search (dense/sparse/hybrid/hybrid_graph) |
| `/paper/{arxiv_id}` | GET | Paper metadata + graph metrics |
| `/citations/{arxiv_id}` | GET | Citation network (cited_by, cites, community) |
| `/graph/stats` | GET | Citation graph statistics |
| `/sessions` | GET | List conversation sessions |
| `/session/{id}` | DELETE | Delete a session |
| `/health` | GET | Health check |

### Gradio UI (`ui/app.py`)

**Technology:** Gradio

**Tabs:**
- **Chat:** Multi-turn conversational Q&A with graph retrieval and reranking toggles
- **Search:** Direct paper search with method selection (dense/sparse/hybrid/hybrid_graph)
- **Paper Details:** Look up paper metadata and citation network by ArXiv ID
- **Citation Graph:** View graph statistics and top papers by PageRank

### Docker Deployment

**Technology:** Docker + Docker Compose

**Services:**
- `api`: FastAPI backend on port 8000 with health checks
- `ui`: Gradio frontend on port 7860, depends on healthy API

**Deployment script** (`scripts/deploy-oracle.sh`): Automates setup on an Ubuntu VM (Oracle Cloud or any provider) - installs Docker, clones the repo, configures firewall, and starts containers.

## Technology Stack Summary

| Component | Technology | Role | Cost |
|-----------|-----------|------|------|
| Paper Collection | arxiv Python library | Download papers from ArXiv API | Free |
| PDF Extraction | PyMuPDF (fitz) | Extract text and sections from PDFs | Free |
| Citation Parsing | Regex + fuzzy matching | Build citation edges between papers | Free |
| Hierarchical Chunking | Custom | Split papers into section/paragraph chunks | Free |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) | 384-dim dense vectors for semantic search | Free |
| Vector Database | ChromaDB | Persistent vector storage with cosine similarity | Free |
| Sparse Search | rank-bm25 | BM25 Okapi keyword matching | Free |
| Citation Graph | NetworkX | Directed graph with PageRank and Louvain communities | Free |
| Cross-Encoder | ms-marco-MiniLM-L-6-v2 | Passage reranking for relevance refinement | Free |
| LLM Generation | Groq API (Llama 3.3 70B) | Answer generation with citations | Free tier |
| Session Storage | SQLite | Conversation persistence | Free |
| API Framework | FastAPI + Uvicorn | REST API with async support | Free |
| Web Interface | Gradio | Chat UI, search, paper details, graph stats | Free |
| Containerization | Docker + Docker Compose | Deployment packaging | Free |
| Environment Config | python-dotenv | Load API keys from .env files | Free |

**Total cost: $0/month** (all local or free-tier services)
