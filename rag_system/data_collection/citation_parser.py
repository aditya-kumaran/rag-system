import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

ARXIV_ID_PATTERN = re.compile(r"(\d{4}\.\d{4,5}(?:v\d+)?)")
DOI_PATTERN = re.compile(r"(10\.\d{4,}/[^\s,]+)")

TITLE_PATTERNS = [
    re.compile(r'["\u201c]([^"\u201d]{15,200})["\u201d]'),
    re.compile(r"(?:^|\n)\s*\[?\d+\]?\s*[A-Z][^.]+?\.\s+(.{15,200}?)\.\s+(?:In\s|Proceedings|arXiv|Journal|Trans)", re.MULTILINE),
]


@dataclass
class ParsedReference:
    raw_text: str
    arxiv_id: Optional[str] = None
    doi: Optional[str] = None
    title: Optional[str] = None
    authors: list[str] = field(default_factory=list)
    year: Optional[int] = None
    venue: Optional[str] = None


@dataclass
class CitationData:
    arxiv_id: str
    references: list[ParsedReference] = field(default_factory=list)
    cited_arxiv_ids: list[str] = field(default_factory=list)
    cited_titles: list[str] = field(default_factory=list)


class CitationParser:
    def __init__(self, known_papers: Optional[dict] = None):
        self.known_papers = known_papers or {}
        self._title_to_id: dict[str, str] = {}
        if self.known_papers:
            for aid, meta in self.known_papers.items():
                title = meta.get("title", "") if isinstance(meta, dict) else getattr(meta, "title", "")
                if title:
                    normalized = self._normalize_title(title)
                    self._title_to_id[normalized] = aid

    def _normalize_title(self, title: str) -> str:
        title = title.lower().strip()
        title = re.sub(r"[^a-z0-9\s]", "", title)
        title = re.sub(r"\s+", " ", title)
        return title

    def parse_references(self, references_text: str, source_arxiv_id: str) -> CitationData:
        if not references_text or not references_text.strip():
            return CitationData(arxiv_id=source_arxiv_id)

        ref_blocks = self._split_references(references_text)
        parsed_refs: list[ParsedReference] = []
        cited_ids: list[str] = []
        cited_titles: list[str] = []

        for block in ref_blocks:
            ref = self._parse_single_reference(block)
            parsed_refs.append(ref)

            if ref.arxiv_id and ref.arxiv_id != source_arxiv_id:
                base_id = ref.arxiv_id.split("v")[0] if "v" in ref.arxiv_id else ref.arxiv_id
                cited_ids.append(base_id)

            if ref.title:
                normalized = self._normalize_title(ref.title)
                if normalized in self._title_to_id:
                    matched_id = self._title_to_id[normalized]
                    if matched_id != source_arxiv_id and matched_id not in cited_ids:
                        cited_ids.append(matched_id)
                cited_titles.append(ref.title)

        return CitationData(
            arxiv_id=source_arxiv_id,
            references=parsed_refs,
            cited_arxiv_ids=cited_ids,
            cited_titles=cited_titles,
        )

    def _split_references(self, text: str) -> list[str]:
        numbered = re.split(r"\n\s*\[(\d+)\]", text)
        if len(numbered) > 3:
            blocks = []
            for i in range(1, len(numbered), 2):
                if i + 1 < len(numbered):
                    blocks.append(f"[{numbered[i]}] {numbered[i+1].strip()}")
            if blocks:
                return blocks

        author_split = re.split(r"\n(?=[A-Z][a-z]+,?\s+[A-Z])", text)
        if len(author_split) > 3:
            return [b.strip() for b in author_split if b.strip()]

        paragraphs = re.split(r"\n\s*\n", text)
        if len(paragraphs) > 2:
            return [p.strip() for p in paragraphs if p.strip()]

        lines = text.split("\n")
        return [line.strip() for line in lines if len(line.strip()) > 20]

    def _parse_single_reference(self, text: str) -> ParsedReference:
        ref = ParsedReference(raw_text=text)

        arxiv_match = ARXIV_ID_PATTERN.search(text)
        if arxiv_match:
            ref.arxiv_id = arxiv_match.group(1)

        doi_match = DOI_PATTERN.search(text)
        if doi_match:
            ref.doi = doi_match.group(1).rstrip(".")

        for pattern in TITLE_PATTERNS:
            match = pattern.search(text)
            if match:
                ref.title = match.group(1).strip().rstrip(".")
                break

        year_match = re.search(r"\b(20[012]\d)\b", text)
        if year_match:
            ref.year = int(year_match.group(1))

        return ref

    def parse_all_papers(
        self,
        extracted_papers: dict,
        output_path: Optional[str | Path] = None,
    ) -> dict[str, CitationData]:
        all_citations: dict[str, CitationData] = {}

        for arxiv_id, paper in extracted_papers.items():
            refs_text = ""
            if isinstance(paper, dict):
                refs_text = paper.get("references_text", "")
            else:
                refs_text = getattr(paper, "references_text", "")

            citation_data = self.parse_references(refs_text, arxiv_id)
            all_citations[arxiv_id] = citation_data

            if citation_data.cited_arxiv_ids:
                logger.info(
                    f"{arxiv_id}: found {len(citation_data.cited_arxiv_ids)} ArXiv citations, "
                    f"{len(citation_data.references)} total refs"
                )

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            save_data = {}
            for aid, cd in all_citations.items():
                save_data[aid] = {
                    "arxiv_id": cd.arxiv_id,
                    "cited_arxiv_ids": cd.cited_arxiv_ids,
                    "cited_titles": cd.cited_titles,
                    "num_references": len(cd.references),
                }
            with open(output_path, "w") as f:
                json.dump(save_data, f, indent=2)
            logger.info(f"Saved citation data to {output_path}")

        total_edges = sum(len(cd.cited_arxiv_ids) for cd in all_citations.values())
        logger.info(
            f"Parsed citations for {len(all_citations)} papers, "
            f"{total_edges} total citation edges"
        )

        return all_citations
