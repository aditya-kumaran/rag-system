import json
import logging

import gradio as gr

logger = logging.getLogger(__name__)

_pipeline = None


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        from rag_system.pipeline import RAGPipeline

        _pipeline = RAGPipeline.load_from_disk()
    return _pipeline


def _format_chat_history(history):
    if not history:
        return "*No messages yet. Ask a question to get started.*"
    md = ""
    for turn in history:
        md += f"**You:** {turn[0]}\n\n**Assistant:** {turn[1]}\n\n---\n\n"
    return md


def chat_response(message, history_json, conversation_id, use_graph, use_reranking):
    pipeline = get_pipeline()
    history = json.loads(history_json) if history_json else []

    if pipeline is None:
        err = "System not initialized. Please run the indexing pipeline first."
        history.append([message, err])
        return _format_chat_history(history), json.dumps(history), conversation_id, ""

    try:
        result = pipeline.query(
            query=message,
            conversation_id=conversation_id if conversation_id else None,
            top_k=10,
            use_graph=use_graph,
            use_reranking=use_reranking,
        )

        answer = result.get("answer", "No answer generated.")
        conv_id = result.get("conversation_id", conversation_id)

        papers_info = ""
        seen = set()
        for r in result.get("retrieved", []):
            aid = r.arxiv_id if hasattr(r, "arxiv_id") else r.get("arxiv_id", "")
            title = r.paper_title if hasattr(r, "paper_title") else r.get("paper_title", "")
            score = r.score if hasattr(r, "score") else r.get("score", 0)

            if aid and aid not in seen:
                seen.add(aid)
                papers_info += f"- **[{aid}]** {title} (score: {score:.3f})\n"

        history.append([message, answer])
        return _format_chat_history(history), json.dumps(history), conv_id, papers_info

    except Exception as e:
        logger.error(f"Chat error: {e}")
        history.append([message, f"Error: {e}"])
        return _format_chat_history(history), json.dumps(history), conversation_id, ""


def search_papers(query, method, top_k):
    pipeline = get_pipeline()
    if pipeline is None:
        return "System not initialized."

    try:
        results = pipeline.search(query=query, top_k=int(top_k), method=method)

        output = f"### Search Results ({len(results)} found, method: {method})\n\n"
        seen = set()
        for r in results:
            aid = r.arxiv_id if hasattr(r, "arxiv_id") else r.get("arxiv_id", "")
            title = r.paper_title if hasattr(r, "paper_title") else r.get("paper_title", "")
            score = r.score if hasattr(r, "score") else r.get("score", 0)
            section = r.section_name if hasattr(r, "section_name") else r.get("section_name", "")
            text = r.text if hasattr(r, "text") else r.get("text", "")

            if aid not in seen:
                seen.add(aid)
                output += f"**[{aid}] {title}**\n"
                output += f"Section: {section} | Score: {score:.4f}\n"
                output += f"{text[:300]}...\n\n---\n\n"

        return output

    except Exception as e:
        return f"Error: {str(e)}"


def get_paper_info(arxiv_id):
    pipeline = get_pipeline()
    if pipeline is None:
        return "System not initialized.", ""

    try:
        paper = pipeline.get_paper(arxiv_id)
        if paper is None:
            return f"Paper {arxiv_id} not found.", ""

        info = f"### {paper['title']}\n\n"
        info += f"**ArXiv ID:** {paper['arxiv_id']}\n\n"
        info += f"**Authors:** {', '.join(paper.get('authors', []))}\n\n"
        info += f"**Published:** {paper.get('published', 'N/A')}\n\n"
        info += f"**Categories:** {', '.join(paper.get('categories', []))}\n\n"
        info += f"**Citation Count:** {paper.get('citation_count', 0)}\n\n"
        info += f"**PageRank:** {paper.get('pagerank_score', 0):.6f}\n\n"
        info += f"**Abstract:**\n{paper.get('abstract', 'N/A')}\n"

        citations = pipeline.get_citations(arxiv_id)
        citation_info = ""
        if citations:
            citation_info = f"### Citation Network for {arxiv_id}\n\n"
            citation_info += f"**Cited by ({len(citations.get('cited_by', []))}):**\n"
            for p in citations.get("cited_by", [])[:10]:
                citation_info += f"- [{p['arxiv_id']}] {p['title']}\n"

            citation_info += f"\n**Cites ({len(citations.get('cites', []))}):**\n"
            for p in citations.get("cites", [])[:10]:
                citation_info += f"- [{p['arxiv_id']}] {p['title']}\n"

            citation_info += f"\n**Same Community ({len(citations.get('community_members', []))}):**\n"
            for p in citations.get("community_members", [])[:10]:
                citation_info += f"- [{p['arxiv_id']}] {p['title']}\n"

        return info, citation_info

    except Exception as e:
        return f"Error: {str(e)}", ""


def get_graph_stats():
    pipeline = get_pipeline()
    if pipeline is None:
        return "System not initialized."

    try:
        stats = pipeline.get_graph_stats()
        stats_text = "### Citation Graph Statistics\n\n"
        stats_text += f"- **Nodes:** {stats.get('nodes', 0)}\n"
        stats_text += f"- **Edges:** {stats.get('edges', 0)}\n"
        stats_text += f"- **Density:** {stats.get('density', 0):.4f}\n"
        stats_text += f"- **Communities:** {stats.get('num_communities', 0)}\n"
        stats_text += f"- **Connected Components:** {stats.get('connected_components', 0)}\n"
        stats_text += f"- **Largest Component:** {stats.get('largest_component_size', 0)}\n"
        stats_text += f"- **Avg In-Degree:** {stats.get('avg_in_degree', 0):.2f}\n"
        stats_text += f"- **Max In-Degree:** {stats.get('max_in_degree', 0)}\n"

        top = stats.get("top_papers_by_pagerank", [])
        if top:
            stats_text += "\n### Top Papers by PageRank\n\n"
            for p in top[:15]:
                if isinstance(p, dict):
                    stats_text += (
                        f"- **[{p.get('arxiv_id', '')}]** "
                        f"{p.get('title', 'N/A')} "
                        f"(PageRank: {p.get('pagerank', 0):.6f})\n"
                    )

        return stats_text

    except Exception as e:
        return f"Error: {str(e)}"


def create_ui():
    with gr.Blocks(
        title="RAG System - ArXiv Paper Q&A",
        theme=gr.themes.Soft(),
        analytics_enabled=False,
    ) as demo:
        gr.Markdown("# RAG System - ArXiv Paper Q&A")
        gr.Markdown(
            "Ask questions about retrieval-augmented generation, dense retrieval, and NLP papers."
        )

        with gr.Tabs():
            with gr.Tab("Chat"):
                chat_display = gr.Markdown(
                    value="*No messages yet. Ask a question to get started.*",
                )
                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder="Ask about RAG, dense retrieval, or NLP papers...",
                        scale=4,
                        show_label=False,
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1)

                with gr.Row():
                    use_graph = gr.Checkbox(value=True, label="Use Graph Retrieval")
                    use_reranking = gr.Checkbox(value=True, label="Use Reranking")
                    conv_id = gr.Textbox(label="Conversation ID", interactive=False)

                history_state = gr.Textbox(value="[]", visible=False)
                papers_display = gr.Markdown(value="*Retrieved papers will appear here*")

                send_btn.click(
                    chat_response,
                    [msg_input, history_state, conv_id, use_graph, use_reranking],
                    [chat_display, history_state, conv_id, papers_display],
                )
                msg_input.submit(
                    chat_response,
                    [msg_input, history_state, conv_id, use_graph, use_reranking],
                    [chat_display, history_state, conv_id, papers_display],
                )

            with gr.Tab("Search"):
                with gr.Row():
                    search_input = gr.Textbox(
                        placeholder="Search papers...",
                        scale=3,
                        show_label=False,
                    )
                    search_method = gr.Dropdown(
                        choices=["hybrid", "dense", "sparse", "hybrid_graph"],
                        value="hybrid",
                        label="Method",
                        scale=1,
                    )
                    search_k = gr.Slider(5, 50, value=20, step=5, label="Top K")
                    search_btn = gr.Button("Search", variant="primary", scale=1)

                search_results = gr.Markdown(value="*Search results will appear here*")

                search_btn.click(
                    search_papers,
                    [search_input, search_method, search_k],
                    search_results,
                )

            with gr.Tab("Paper Details"):
                with gr.Row():
                    paper_id_input = gr.Textbox(
                        placeholder="Enter ArXiv ID (e.g., 2004.04906)",
                        label="ArXiv ID",
                        scale=3,
                    )
                    paper_btn = gr.Button("Look Up", variant="primary", scale=1)

                paper_info = gr.Markdown(value="*Paper details will appear here*")
                citation_info = gr.Markdown(value="*Citation network will appear here*")

                paper_btn.click(
                    get_paper_info,
                    paper_id_input,
                    [paper_info, citation_info],
                )

            with gr.Tab("Citation Graph"):
                graph_btn = gr.Button("Load Graph Statistics", variant="primary")
                graph_stats_display = gr.Markdown(value="*Click button to see statistics*")

                graph_btn.click(
                    get_graph_stats,
                    [],
                    graph_stats_display,
                )

    return demo


def main():
    logging.basicConfig(level=logging.INFO)
    demo = create_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, show_api=False)


if __name__ == "__main__":
    main()
