import logging

from rag_system.conversation.session_manager import SessionState

logger = logging.getLogger(__name__)

MAX_RECENT_VERBATIM = 5
MAX_CONTEXT_TOKENS = 2000


class ContextCompressor:
    def __init__(self, llm_client=None):
        self.llm_client = llm_client

    def compress(
        self,
        session: SessionState,
        max_recent: int = MAX_RECENT_VERBATIM,
    ) -> str:
        if len(session.messages) <= max_recent:
            return session.get_conversation_text()

        older_messages = session.messages[:-max_recent]
        recent_messages = session.messages[-max_recent:]

        compressed = self._compress_messages(older_messages, session)

        recent_text = ""
        for msg in recent_messages:
            recent_text += f"{msg.role}: {msg.content}\n"

        full_context = ""
        if compressed:
            full_context = f"[Previous context summary]\n{compressed}\n\n"
        full_context += f"[Recent messages]\n{recent_text}"

        session.compressed_context = compressed
        return full_context

    def _compress_messages(
        self,
        messages: list,
        session: SessionState,
    ) -> str:
        if self.llm_client:
            return self._llm_compress(messages, session)
        return self._extractive_compress(messages, session)

    def _extractive_compress(
        self,
        messages: list,
        session: SessionState,
    ) -> str:
        topics = set()
        papers_mentioned = set()
        key_points = []

        for msg in messages:
            content = msg.content

            for short, full in session.entities.items():
                if short.lower() in content.lower():
                    topics.add(full)

            for arxiv_id in session.papers_discussed:
                if arxiv_id in content:
                    papers_mentioned.add(arxiv_id)

            if msg.role == "user":
                key_points.append(f"Q: {content[:150]}")
            elif msg.role == "assistant" and len(content) > 100:
                sentences = content.split(".")
                if sentences:
                    key_points.append(f"A: {sentences[0].strip()}")

        summary_parts = []

        if topics:
            summary_parts.append(f"Topics discussed: {', '.join(list(topics)[:10])}")

        if papers_mentioned:
            summary_parts.append(f"Papers referenced: {', '.join(list(papers_mentioned)[:10])}")

        if key_points:
            summary_parts.append("Key exchanges:")
            for point in key_points[-5:]:
                summary_parts.append(f"  - {point}")

        return "\n".join(summary_parts)

    def _llm_compress(
        self,
        messages: list,
        session: SessionState,
    ) -> str:
        from rag_system.generation.prompt_templates import CONTEXT_COMPRESSION_TEMPLATE

        conversation_text = ""
        for msg in messages:
            conversation_text += f"{msg.role}: {msg.content}\n"

        if len(conversation_text) > 4000:
            conversation_text = conversation_text[:4000] + "..."

        prompt = CONTEXT_COMPRESSION_TEMPLATE.format(conversation=conversation_text)

        try:
            if hasattr(self.llm_client, "client"):
                response = self.llm_client.client.chat.completions.create(
                    model=self.llm_client.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                    max_tokens=512,
                )
                compressed = response.choices[0].message.content.strip()
                if compressed:
                    logger.info(f"Compressed {len(messages)} messages into {len(compressed)} chars")
                    return compressed
        except Exception as e:
            logger.warning(f"LLM compression failed: {e}")

        return self._extractive_compress(messages, session)

    def get_context_for_retrieval(
        self,
        session: SessionState,
        current_query: str,
    ) -> dict:
        return {
            "query": current_query,
            "papers_discussed": session.papers_discussed.copy(),
            "entities": session.entities.copy(),
            "recent_topics": self._extract_recent_topics(session),
        }

    def _extract_recent_topics(self, session: SessionState) -> list[str]:
        topics = []
        for msg in session.messages[-4:]:
            if msg.role == "user":
                words = msg.content.split()
                for word in words:
                    if len(word) > 4 and word[0].isupper():
                        topics.append(word)
        return list(set(topics))[:5]
