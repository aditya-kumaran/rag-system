SYSTEM_PROMPT = """You are an expert AI research assistant specializing in scientific papers about retrieval-augmented generation (RAG), information retrieval, and natural language processing.

Guidelines:
1. Always cite papers using their ArXiv IDs (e.g., [2004.04906]) when referencing specific findings
2. Prefer information from more recent papers (2023-2025) unless historical context is needed
3. When comparing methods, provide specific metrics and benchmarks mentioned in the papers
4. Acknowledge uncertainty when information is incomplete or conflicting across papers
5. Structure responses clearly with sections when answering survey or comparative questions
6. If the retrieved context doesn't contain enough information, say so honestly
7. In a multi-turn conversation, DO NOT repeat or re-explain information you already provided in earlier turns. Build on what was said before. Reference it briefly if needed but focus on new information.
8. For follow-up questions, answer directly and concisely without restating the background or context from previous answers."""

QUERY_TEMPLATE = """Based on the following retrieved papers and context, answer the user's question.

Retrieved Context:
{context}

Papers Referenced:
{papers_list}

User Question: {query}

Provide a comprehensive answer citing specific papers by their ArXiv IDs. If the context doesn't fully address the question, acknowledge what's missing."""

COMPARATIVE_TEMPLATE = """Based on the following retrieved papers, provide a detailed comparison.

Retrieved Context:
{context}

Papers Referenced:
{papers_list}

Comparison Request: {query}

Structure your response as:
1. Brief overview of each method/approach
2. Key differences (architecture, training, data requirements)
3. Performance comparison (cite specific metrics from papers)
4. Strengths and weaknesses of each
5. Recommendation based on use case"""

SURVEY_TEMPLATE = """Based on the following retrieved papers, provide a comprehensive survey response.

Retrieved Context:
{context}

Papers Referenced:
{papers_list}

Survey Question: {query}

Structure your response as:
1. Overview of the research area
2. Key developments organized chronologically or thematically
3. Current state-of-the-art approaches
4. Open challenges and future directions
5. Most influential papers in this area"""

HYDE_TEMPLATE = """Write a short, technical passage that would answer the following question about AI/ML research. Write as if it's an excerpt from an academic paper. Be specific and use technical terminology.

Question: {query}

Hypothetical answer passage:"""

CONTEXT_COMPRESSION_TEMPLATE = """Summarize the following conversation history into a concise context that captures:
1. The main topics discussed
2. Key papers and methods mentioned
3. Any specific findings or conclusions reached
4. The user's apparent research interests

Conversation:
{conversation}

Compressed context (be concise but preserve all important details):"""

COREFERENCE_TEMPLATE = """Given the conversation history and the current query, rewrite the query to be self-contained by resolving any references to previous messages.

Conversation History:
{history}

Current Query: {query}

Rewritten Query (self-contained, no pronouns or references to previous messages):"""


def format_context(results: list, max_chars: int = 8000) -> str:
    context_parts = []
    total_chars = 0

    for result in results:
        if isinstance(result, dict):
            arxiv_id = result.get("arxiv_id", "")
            title = result.get("paper_title", "")
            section = result.get("section_name", "")
            text = result.get("text", "")
        else:
            arxiv_id = getattr(result, "arxiv_id", "")
            title = getattr(result, "paper_title", "")
            section = getattr(result, "section_name", "")
            text = getattr(result, "text", "")

        entry = f"[{arxiv_id}] {title}"
        if section:
            entry += f" - {section}"
        entry += f"\n{text}\n"

        if total_chars + len(entry) > max_chars:
            remaining = max_chars - total_chars
            if remaining > 200:
                entry = entry[:remaining] + "..."
                context_parts.append(entry)
            break

        context_parts.append(entry)
        total_chars += len(entry)

    return "\n---\n".join(context_parts)


def format_papers_list(results: list) -> str:
    seen = set()
    papers = []

    for result in results:
        if isinstance(result, dict):
            arxiv_id = result.get("arxiv_id", "")
            title = result.get("paper_title", "")
        else:
            arxiv_id = getattr(result, "arxiv_id", "")
            title = getattr(result, "paper_title", "")

        if arxiv_id and arxiv_id not in seen:
            seen.add(arxiv_id)
            papers.append(f"- [{arxiv_id}] {title}")

    return "\n".join(papers)


def get_prompt_for_query_type(query_type: str) -> str:
    templates = {
        "comparative": COMPARATIVE_TEMPLATE,
        "survey": SURVEY_TEMPLATE,
    }
    return templates.get(query_type, QUERY_TEMPLATE)
