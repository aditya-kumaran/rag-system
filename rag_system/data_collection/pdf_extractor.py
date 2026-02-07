import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import fitz

logger = logging.getLogger(__name__)

SECTION_PATTERNS = [
    r"^(?:\d+\.?\s+)?(abstract)\s*$",
    r"^(?:\d+\.?\s+)?(introduction)\s*$",
    r"^(?:\d+\.?\s+)?(related\s+work)\s*$",
    r"^(?:\d+\.?\s+)?(background)\s*$",
    r"^(?:\d+\.?\s+)?(method(?:s|ology)?)\s*$",
    r"^(?:\d+\.?\s+)?(approach)\s*$",
    r"^(?:\d+\.?\s+)?(model(?:\s+architecture)?)\s*$",
    r"^(?:\d+\.?\s+)?(experiment(?:s|al\s+(?:setup|results))?)\s*$",
    r"^(?:\d+\.?\s+)?(results?(?:\s+and\s+(?:discussion|analysis))?)\s*$",
    r"^(?:\d+\.?\s+)?(discussion)\s*$",
    r"^(?:\d+\.?\s+)?(analysis)\s*$",
    r"^(?:\d+\.?\s+)?(evaluation)\s*$",
    r"^(?:\d+\.?\s+)?(conclusion(?:s)?)\s*$",
    r"^(?:\d+\.?\s+)?(future\s+work)\s*$",
    r"^(?:\d+\.?\s+)?(limitations?)\s*$",
    r"^(?:\d+\.?\s+)?(references?|bibliography)\s*$",
    r"^(?:\d+\.?\s+)?(appendix(?:\s+[a-z])?)\s*$",
    r"^(?:\d+\.?\s+)?(acknowledgement(?:s)?)\s*$",
]

SECTION_COMPILED = [re.compile(p, re.IGNORECASE) for p in SECTION_PATTERNS]

NUMBERED_SECTION = re.compile(r"^(\d+(?:\.\d+)*)\s+(.+)$")


@dataclass
class Section:
    name: str
    text: str
    start_page: int
    end_page: int
    subsections: list["Section"] = field(default_factory=list)


@dataclass
class ExtractedPaper:
    arxiv_id: str
    title: str
    full_text: str
    abstract: str
    sections: list[Section]
    references_text: str
    num_pages: int
    metadata: dict = field(default_factory=dict)


class PDFExtractor:
    def __init__(self, min_section_length: int = 50):
        self.min_section_length = min_section_length

    def extract(self, pdf_path: str | Path, arxiv_id: str = "") -> Optional[ExtractedPaper]:
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            logger.error(f"PDF not found: {pdf_path}")
            return None

        try:
            doc = fitz.open(str(pdf_path))
        except Exception as e:
            logger.error(f"Failed to open PDF {pdf_path}: {e}")
            return None

        if not arxiv_id:
            arxiv_id = pdf_path.stem

        try:
            full_text = self._extract_full_text(doc)
            title = self._extract_title(doc, full_text)
            abstract = self._extract_abstract(full_text)
            sections = self._extract_sections(doc, full_text)
            references_text = self._extract_references(full_text)

            paper = ExtractedPaper(
                arxiv_id=arxiv_id,
                title=title,
                full_text=full_text,
                abstract=abstract,
                sections=sections,
                references_text=references_text,
                num_pages=len(doc),
                metadata={
                    "pdf_path": str(pdf_path),
                    "file_size_kb": pdf_path.stat().st_size / 1024,
                },
            )

            doc.close()
            return paper

        except Exception as e:
            logger.error(f"Error extracting {arxiv_id}: {e}")
            doc.close()
            return None

    def _extract_full_text(self, doc: fitz.Document) -> str:
        pages = []
        for page in doc:
            text = page.get_text("text")
            if text.strip():
                pages.append(text)
        return "\n\n".join(pages)

    def _extract_title(self, doc: fitz.Document, full_text: str) -> str:
        if len(doc) > 0:
            first_page = doc[0]
            blocks = first_page.get_text("dict")["blocks"]
            max_size = 0
            title_text = ""

            for block in blocks:
                if "lines" not in block:
                    continue
                for line in block["lines"]:
                    for span in line["spans"]:
                        if span["size"] > max_size and len(span["text"].strip()) > 5:
                            max_size = span["size"]
                            title_text = span["text"].strip()

            if title_text:
                for block in blocks:
                    if "lines" not in block:
                        continue
                    for line in block["lines"]:
                        for span in line["spans"]:
                            if abs(span["size"] - max_size) < 0.5 and span["text"].strip() != title_text:
                                title_text += " " + span["text"].strip()
                                break

            if title_text and len(title_text) > 10:
                return title_text.replace("\n", " ").strip()

        lines = full_text.split("\n")
        for line in lines[:20]:
            line = line.strip()
            if len(line) > 15 and not line.startswith("arXiv"):
                return line

        return "Unknown Title"

    def _extract_abstract(self, full_text: str) -> str:
        patterns = [
            r"(?:Abstract|ABSTRACT)[.\s\-—]*\n(.*?)(?:\n\s*\n|\n(?:\d+\.?\s+)?(?:Introduction|INTRODUCTION))",
            r"(?:Abstract|ABSTRACT)[.\s\-—]*(.*?)(?:(?:\d+\.?\s+)?Introduction|Keywords|1\s)",
        ]

        for pattern in patterns:
            match = re.search(pattern, full_text, re.DOTALL | re.IGNORECASE)
            if match:
                abstract = match.group(1).strip()
                abstract = re.sub(r"\s+", " ", abstract)
                if len(abstract) > 50:
                    return abstract

        lines = full_text.split("\n")
        for i, line in enumerate(lines[:50]):
            if "abstract" in line.lower():
                abstract_lines = []
                for j in range(i + 1, min(i + 30, len(lines))):
                    if lines[j].strip() == "" and abstract_lines:
                        break
                    if re.match(r"^\d+\.?\s+(Introduction|INTRODUCTION)", lines[j]):
                        break
                    if lines[j].strip():
                        abstract_lines.append(lines[j].strip())
                if abstract_lines:
                    return " ".join(abstract_lines)

        return ""

    def _extract_sections(self, doc: fitz.Document, full_text: str) -> list[Section]:
        lines = full_text.split("\n")
        section_boundaries: list[tuple[int, str]] = []

        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped or len(stripped) > 100:
                continue

            for pattern in SECTION_COMPILED:
                if pattern.match(stripped):
                    section_boundaries.append((i, stripped))
                    break
            else:
                match = NUMBERED_SECTION.match(stripped)
                if match and len(match.group(2)) < 80:
                    num = match.group(1)
                    if len(num.split(".")) <= 2:
                        section_boundaries.append((i, stripped))

        sections: list[Section] = []
        for idx, (line_idx, section_name) in enumerate(section_boundaries):
            if idx + 1 < len(section_boundaries):
                end_idx = section_boundaries[idx + 1][0]
            else:
                end_idx = len(lines)

            section_text = "\n".join(lines[line_idx + 1 : end_idx]).strip()

            if len(section_text) >= self.min_section_length:
                sections.append(
                    Section(
                        name=section_name,
                        text=section_text,
                        start_page=0,
                        end_page=0,
                    )
                )

        if not sections and full_text.strip():
            sections.append(
                Section(
                    name="Full Text",
                    text=full_text.strip(),
                    start_page=0,
                    end_page=len(doc) - 1,
                )
            )

        return sections

    def _extract_references(self, full_text: str) -> str:
        patterns = [
            r"(?:\n\s*(?:References?|REFERENCES?|Bibliography|BIBLIOGRAPHY)\s*\n)(.*)",
            r"(?:^(?:References?|REFERENCES?|Bibliography|BIBLIOGRAPHY)\s*$)(.*)",
        ]

        for pattern in patterns:
            match = re.search(pattern, full_text, re.DOTALL | re.MULTILINE)
            if match:
                refs = match.group(1).strip()
                appendix_match = re.search(
                    r"\n\s*(?:Appendix|APPENDIX|Supplementary)", refs
                )
                if appendix_match:
                    refs = refs[: appendix_match.start()]
                return refs

        return ""


def extract_papers_batch(
    papers_metadata: dict,
    pdfs_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, ExtractedPaper]:
    pdfs_dir = Path(pdfs_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    extractor = PDFExtractor()
    extracted: dict[str, ExtractedPaper] = {}

    for arxiv_id, meta in papers_metadata.items():
        pdf_path = pdfs_dir / f"{arxiv_id}.pdf"
        if not pdf_path.exists():
            continue

        output_file = output_dir / f"{arxiv_id}.json"
        if output_file.exists():
            try:
                with open(output_file, "r") as f:
                    data = json.load(f)
                extracted[arxiv_id] = ExtractedPaper(
                    arxiv_id=data["arxiv_id"],
                    title=data["title"],
                    full_text=data["full_text"],
                    abstract=data["abstract"],
                    sections=[
                        Section(**s) for s in data.get("sections", [])
                    ],
                    references_text=data.get("references_text", ""),
                    num_pages=data.get("num_pages", 0),
                    metadata=data.get("metadata", {}),
                )
                continue
            except Exception:
                pass

        paper = extractor.extract(pdf_path, arxiv_id)
        if paper:
            extracted[arxiv_id] = paper
            save_data = {
                "arxiv_id": paper.arxiv_id,
                "title": paper.title,
                "full_text": paper.full_text[:50000],
                "abstract": paper.abstract,
                "sections": [
                    {"name": s.name, "text": s.text[:10000], "start_page": s.start_page, "end_page": s.end_page}
                    for s in paper.sections
                ],
                "references_text": paper.references_text[:10000],
                "num_pages": paper.num_pages,
                "metadata": paper.metadata,
            }
            with open(output_file, "w") as f:
                json.dump(save_data, f, indent=2)

            logger.info(f"Extracted {arxiv_id}: {len(paper.sections)} sections")

    logger.info(f"Extracted {len(extracted)}/{len(papers_metadata)} papers")
    return extracted
