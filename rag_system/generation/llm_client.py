import logging
import os
from typing import Optional

from rag_system.generation.prompt_templates import (
    HYDE_TEMPLATE,
    SYSTEM_PROMPT,
    format_context,
    format_papers_list,
    get_prompt_for_query_type,
)

logger = logging.getLogger(__name__)


class LLMClient:
    def __init__(
        self,
        provider: str = "groq",
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ):
        self.provider = provider
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = None

        if provider == "groq":
            self.model = model or "llama-3.3-70b-versatile"
            self.api_key = api_key or os.getenv("GROQ_API_KEY", "")
        elif provider == "openai":
            self.model = model or "gpt-4o-mini"
            self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        elif provider == "together":
            self.model = model or "meta-llama/Llama-3.3-70B-Instruct-Turbo"
            self.api_key = api_key or os.getenv("TOGETHER_API_KEY", "")
        else:
            self.model = model or "llama-3.3-70b-versatile"
            self.api_key = api_key or ""

    @property
    def client(self):
        if self._client is None:
            if self.provider == "groq":
                from groq import Groq

                self._client = Groq(api_key=self.api_key)
            elif self.provider in ("openai", "together"):
                from openai import OpenAI

                base_url = None
                if self.provider == "together":
                    base_url = "https://api.together.xyz/v1"
                self._client = OpenAI(api_key=self.api_key, base_url=base_url)
            else:
                from groq import Groq

                self._client = Groq(api_key=self.api_key)
        return self._client

    def generate(
        self,
        query: str,
        retrieved_results: list,
        query_type: str = "general",
        conversation_context: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        context = format_context(retrieved_results)
        papers_list = format_papers_list(retrieved_results)

        prompt_template = get_prompt_for_query_type(query_type)
        user_prompt = prompt_template.format(
            context=context,
            papers_list=papers_list,
            query=query,
        )

        if conversation_context:
            user_prompt = f"Previous conversation context:\n{conversation_context}\n\n{user_prompt}"

        sys_prompt = system_prompt or SYSTEM_PROMPT

        try:
            if self.provider == "groq":
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                return response.choices[0].message.content

            else:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                return response.choices[0].message.content

        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return self._fallback_response(query, retrieved_results)

    def generate_hyde(self, query: str) -> str:
        prompt = HYDE_TEMPLATE.format(query=query)

        try:
            if self.provider == "groq":
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.7,
                    max_tokens=256,
                )
                return response.choices[0].message.content
            else:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.7,
                    max_tokens=256,
                )
                return response.choices[0].message.content

        except Exception as e:
            logger.error(f"HyDE generation failed: {e}")
            return query

    def _fallback_response(self, query: str, results: list) -> str:
        if not results:
            return "I couldn't find relevant papers to answer your question. Please try rephrasing."

        response_parts = [f"Based on the retrieved papers, here's what I found regarding: {query}\n"]

        seen_papers = set()
        for result in results[:5]:
            if isinstance(result, dict):
                arxiv_id = result.get("arxiv_id", "")
                title = result.get("paper_title", "")
                text = result.get("text", "")
            else:
                arxiv_id = getattr(result, "arxiv_id", "")
                title = getattr(result, "paper_title", "")
                text = getattr(result, "text", "")

            if arxiv_id in seen_papers:
                continue
            seen_papers.add(arxiv_id)

            response_parts.append(f"\n**[{arxiv_id}] {title}**")
            if text:
                response_parts.append(text[:300] + "...")

        response_parts.append(
            "\n\nNote: This is a fallback response generated without LLM. "
            "Configure an API key for better answers."
        )

        return "\n".join(response_parts)
