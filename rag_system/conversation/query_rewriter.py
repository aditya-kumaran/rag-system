import logging
import re

from rag_system.conversation.session_manager import SessionState

logger = logging.getLogger(__name__)

PRONOUNS = {"it", "its", "this", "that", "these", "those", "they", "them", "their", "he", "she", "his", "her"}
REFERENCE_PATTERNS = [
    r"\b(?:the\s+)?(?:paper|method|approach|model|system|technique|framework|algorithm)\b",
    r"\b(?:the\s+)?(?:same|previous|above|mentioned|discussed)\b",
    r"\b(?:it|its|this|that|these|those)\b",
]

COMPILED_REFS = [re.compile(p, re.IGNORECASE) for p in REFERENCE_PATTERNS]


class QueryRewriter:
    def __init__(self, llm_client=None):
        self.llm_client = llm_client

    def rewrite(
        self,
        query: str,
        session: SessionState,
        use_llm: bool = True,
    ) -> str:
        if not self._needs_rewriting(query, session):
            return query

        if use_llm and self.llm_client:
            return self._llm_rewrite(query, session)

        return self._rule_based_rewrite(query, session)

    def _needs_rewriting(self, query: str, session: SessionState) -> bool:
        if len(session.messages) < 2:
            return False

        query_lower = query.lower()
        query_words = set(query_lower.split())

        if query_words & PRONOUNS:
            return True

        for pattern in COMPILED_REFS:
            if pattern.search(query_lower):
                if not any(w in query_lower for w in ["what is a", "define", "explain what"]):
                    return True

        return False

    def _rule_based_rewrite(self, query: str, session: SessionState) -> str:
        rewritten = query

        last_papers = []
        for msg in reversed(session.messages[-6:]):
            if msg.metadata and "arxiv_ids" in msg.metadata:
                last_papers = msg.metadata["arxiv_ids"]
                break

        last_entities: list[str] = []
        for msg in reversed(session.messages[-4:]):
            if msg.role == "assistant":
                for short, full in session.entities.items():
                    if short.lower() in msg.content.lower():
                        last_entities.append(full)

        if not last_entities:
            for msg in reversed(session.messages[-4:]):
                if msg.role == "user":
                    for word in msg.content.split():
                        if len(word) > 3 and word[0].isupper():
                            last_entities.append(word)
                    break

        for pronoun in ["it", "its", "this method", "this approach", "this paper", "the paper", "the method"]:
            if pronoun in rewritten.lower():
                if last_entities:
                    entity = last_entities[0]
                    rewritten = re.sub(
                        re.escape(pronoun),
                        entity,
                        rewritten,
                        count=1,
                        flags=re.IGNORECASE,
                    )
                    break

        if last_papers and "related" in rewritten.lower():
            paper_ref = ", ".join(last_papers[:3])
            rewritten += f" (related to papers: {paper_ref})"

        if rewritten != query:
            logger.info(f"Rewrote query: '{query}' -> '{rewritten}'")

        return rewritten

    def _llm_rewrite(self, query: str, session: SessionState) -> str:
        from rag_system.generation.prompt_templates import COREFERENCE_TEMPLATE

        history = session.get_conversation_text(n=6)
        prompt = COREFERENCE_TEMPLATE.format(history=history, query=query)

        try:
            if hasattr(self.llm_client, "client"):
                response = self.llm_client.client.chat.completions.create(
                    model=self.llm_client.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=256,
                )
                rewritten = response.choices[0].message.content.strip()

                rewritten = rewritten.replace('"', "").replace("'", "")
                if rewritten.lower().startswith("rewritten query:"):
                    rewritten = rewritten[16:].strip()

                if rewritten and len(rewritten) > 5:
                    logger.info(f"LLM rewrote query: '{query}' -> '{rewritten}'")
                    return rewritten

        except Exception as e:
            logger.warning(f"LLM rewrite failed: {e}")

        return self._rule_based_rewrite(query, session)

    def build_search_query(
        self,
        query: str,
        session: SessionState,
        include_context: bool = True,
    ) -> str:
        rewritten = self.rewrite(query, session)

        if not include_context or len(session.messages) < 2:
            return rewritten

        context_terms = []
        for entity_short, entity_full in session.entities.items():
            if entity_short.lower() not in rewritten.lower():
                context_terms.append(entity_full)

        if context_terms:
            rewritten = f"{rewritten} ({', '.join(context_terms[:3])})"

        return rewritten
