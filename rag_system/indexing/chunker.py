import logging
import re
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    chunk_id: str
    text: str
    arxiv_id: str
    paper_title: str
    section_name: str
    chunk_type: str
    position: int
    parent_chunk_id: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    @property
    def token_estimate(self) -> int:
        return len(self.text.split())


class HierarchicalChunker:
    def __init__(
        self,
        max_chunk_tokens: int = 512,
        overlap_ratio: float = 0.2,
        min_chunk_tokens: int = 50,
    ):
        self.max_chunk_tokens = max_chunk_tokens
        self.overlap_tokens = int(max_chunk_tokens * overlap_ratio)
        self.min_chunk_tokens = min_chunk_tokens

    def chunk_paper(
        self,
        arxiv_id: str,
        title: str,
        abstract: str,
        sections: list[dict],
        authors: Optional[list[str]] = None,
        published: Optional[str] = None,
    ) -> list[Chunk]:
        chunks: list[Chunk] = []
        position = 0

        base_metadata = {
            "authors": authors or [],
            "published": published or "",
        }

        if abstract and len(abstract.split()) >= self.min_chunk_tokens:
            chunks.append(
                Chunk(
                    chunk_id=f"{arxiv_id}_abstract",
                    text=abstract,
                    arxiv_id=arxiv_id,
                    paper_title=title,
                    section_name="Abstract",
                    chunk_type="abstract",
                    position=position,
                    metadata=base_metadata.copy(),
                )
            )
            position += 1

        for section in sections:
            section_name = section.get("name", "Unknown")
            section_text = section.get("text", "")

            if not section_text or len(section_text.split()) < self.min_chunk_tokens:
                continue

            if section_name.lower() in ("references", "reference", "bibliography"):
                continue

            section_chunk_id = f"{arxiv_id}_{self._sanitize_name(section_name)}"
            section_words = section_text.split()

            if len(section_words) <= self.max_chunk_tokens:
                chunks.append(
                    Chunk(
                        chunk_id=section_chunk_id,
                        text=section_text,
                        arxiv_id=arxiv_id,
                        paper_title=title,
                        section_name=section_name,
                        chunk_type="section",
                        position=position,
                        metadata=base_metadata.copy(),
                    )
                )
                position += 1
            else:
                paragraphs = self._split_into_paragraphs(section_text)
                sub_chunks = self._merge_paragraphs(
                    paragraphs, section_chunk_id, arxiv_id, title, section_name, base_metadata
                )

                for i, chunk in enumerate(sub_chunks):
                    chunk.position = position
                    chunk.parent_chunk_id = section_chunk_id
                    position += 1
                    chunks.append(chunk)

        if not chunks:
            full_text = abstract or ""
            for section in sections:
                full_text += "\n\n" + section.get("text", "")

            if full_text.strip():
                text_chunks = self._split_text_with_overlap(full_text.strip())
                for i, text in enumerate(text_chunks):
                    chunks.append(
                        Chunk(
                            chunk_id=f"{arxiv_id}_chunk_{i}",
                            text=text,
                            arxiv_id=arxiv_id,
                            paper_title=title,
                            section_name="Full Text",
                            chunk_type="text",
                            position=i,
                            metadata=base_metadata.copy(),
                        )
                    )

        return chunks

    def _split_into_paragraphs(self, text: str) -> list[str]:
        paragraphs = re.split(r"\n\s*\n", text)
        result = []
        for p in paragraphs:
            p = p.strip()
            if p and len(p.split()) >= 10:
                result.append(p)
        return result if result else [text]

    def _merge_paragraphs(
        self,
        paragraphs: list[str],
        parent_id: str,
        arxiv_id: str,
        title: str,
        section_name: str,
        base_metadata: dict,
    ) -> list[Chunk]:
        chunks: list[Chunk] = []
        current_text = ""
        chunk_idx = 0

        for para in paragraphs:
            para_words = len(para.split())
            current_words = len(current_text.split()) if current_text else 0

            if current_words + para_words > self.max_chunk_tokens and current_text:
                chunks.append(
                    Chunk(
                        chunk_id=f"{parent_id}_p{chunk_idx}",
                        text=current_text.strip(),
                        arxiv_id=arxiv_id,
                        paper_title=title,
                        section_name=section_name,
                        chunk_type="paragraph",
                        position=0,
                        metadata=base_metadata.copy(),
                    )
                )
                chunk_idx += 1

                if self.overlap_tokens > 0:
                    words = current_text.split()
                    overlap_text = " ".join(words[-self.overlap_tokens :])
                    current_text = overlap_text + "\n\n" + para
                else:
                    current_text = para
            else:
                current_text = (current_text + "\n\n" + para).strip() if current_text else para

        if current_text and len(current_text.split()) >= self.min_chunk_tokens:
            chunks.append(
                Chunk(
                    chunk_id=f"{parent_id}_p{chunk_idx}",
                    text=current_text.strip(),
                    arxiv_id=arxiv_id,
                    paper_title=title,
                    section_name=section_name,
                    chunk_type="paragraph",
                    position=0,
                    metadata=base_metadata.copy(),
                )
            )

        return chunks

    def _split_text_with_overlap(self, text: str) -> list[str]:
        words = text.split()
        if len(words) <= self.max_chunk_tokens:
            return [text]

        chunks = []
        start = 0
        while start < len(words):
            end = min(start + self.max_chunk_tokens, len(words))
            chunk_text = " ".join(words[start:end])
            chunks.append(chunk_text)
            start = end - self.overlap_tokens

        return chunks

    def _sanitize_name(self, name: str) -> str:
        name = re.sub(r"[^a-zA-Z0-9\s]", "", name)
        name = re.sub(r"\s+", "_", name.strip())
        return name.lower()[:50]


def chunk_all_papers(
    extracted_papers: dict,
    papers_metadata: dict,
    chunker: Optional[HierarchicalChunker] = None,
) -> list[Chunk]:
    if chunker is None:
        chunker = HierarchicalChunker()

    all_chunks: list[Chunk] = []

    for arxiv_id, paper in extracted_papers.items():
        if isinstance(paper, dict):
            title = paper.get("title", "")
            abstract = paper.get("abstract", "")
            sections = paper.get("sections", [])
        else:
            title = getattr(paper, "title", "")
            abstract = getattr(paper, "abstract", "")
            sections_raw = getattr(paper, "sections", [])
            sections = []
            for s in sections_raw:
                if isinstance(s, dict):
                    sections.append(s)
                else:
                    sections.append({"name": getattr(s, "name", ""), "text": getattr(s, "text", "")})

        meta = papers_metadata.get(arxiv_id, {})
        if isinstance(meta, dict):
            authors = meta.get("authors", [])
            published = meta.get("published", "")
        else:
            authors = getattr(meta, "authors", [])
            published = getattr(meta, "published", "")

        paper_chunks = chunker.chunk_paper(
            arxiv_id=arxiv_id,
            title=title,
            abstract=abstract,
            sections=sections,
            authors=authors,
            published=published,
        )
        all_chunks.extend(paper_chunks)

    logger.info(f"Created {len(all_chunks)} chunks from {len(extracted_papers)} papers")
    return all_chunks
