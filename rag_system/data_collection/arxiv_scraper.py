import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import arxiv
import httpx

logger = logging.getLogger(__name__)

SEARCH_QUERIES = [
    "retrieval augmented generation",
    "dense retrieval",
    "neural information retrieval",
    "question answering retrieval",
    "RAG language model",
    "dense passage retrieval",
    "semantic search neural",
    "knowledge grounded generation",
    "open domain question answering",
    "document retrieval transformer",
]

DEFAULT_DATA_DIR = Path("data")
DEFAULT_PAPERS_DIR = DEFAULT_DATA_DIR / "papers"
DEFAULT_PDFS_DIR = DEFAULT_DATA_DIR / "pdfs"
DEFAULT_METADATA_FILE = DEFAULT_DATA_DIR / "papers_metadata.json"


@dataclass
class PaperMetadata:
    arxiv_id: str
    title: str
    authors: list[str]
    abstract: str
    published: str
    updated: str
    categories: list[str]
    primary_category: str
    pdf_url: str
    comment: Optional[str] = None
    journal_ref: Optional[str] = None
    doi: Optional[str] = None
    pdf_path: Optional[str] = None
    text_extracted: bool = False
    citations_parsed: bool = False
    references: list[str] = field(default_factory=list)


class ArxivScraper:
    def __init__(
        self,
        data_dir: str | Path = DEFAULT_DATA_DIR,
        max_papers: int = 500,
        min_year: int = 2020,
        max_year: int = 2025,
    ):
        self.data_dir = Path(data_dir)
        self.papers_dir = self.data_dir / "papers"
        self.pdfs_dir = self.data_dir / "pdfs"
        self.metadata_file = self.data_dir / "papers_metadata.json"
        self.max_papers = max_papers
        self.min_year = min_year
        self.max_year = max_year
        self.papers: dict[str, PaperMetadata] = {}

        self.papers_dir.mkdir(parents=True, exist_ok=True)
        self.pdfs_dir.mkdir(parents=True, exist_ok=True)

    def search_papers(self) -> dict[str, PaperMetadata]:
        seen_ids: set[str] = set()
        papers_per_query = (self.max_papers // len(SEARCH_QUERIES)) + 10

        for query in SEARCH_QUERIES:
            if len(seen_ids) >= self.max_papers:
                break

            logger.info(f"Searching: '{query}' (have {len(seen_ids)} papers)")
            try:
                search = arxiv.Search(
                    query=query,
                    max_results=papers_per_query,
                    sort_by=arxiv.SortCriterion.Relevance,
                )
                client = arxiv.Client(
                    page_size=50,
                    delay_seconds=3.0,
                    num_retries=3,
                )

                for result in client.results(search):
                    if len(seen_ids) >= self.max_papers:
                        break

                    arxiv_id = result.entry_id.split("/")[-1]
                    arxiv_id = arxiv_id.replace("v1", "").replace("v2", "").replace("v3", "")
                    if "v" in arxiv_id:
                        arxiv_id = arxiv_id.split("v")[0]

                    if arxiv_id in seen_ids:
                        continue

                    pub_year = result.published.year
                    if pub_year < self.min_year or pub_year > self.max_year:
                        continue

                    paper = PaperMetadata(
                        arxiv_id=arxiv_id,
                        title=result.title.replace("\n", " ").strip(),
                        authors=[a.name for a in result.authors],
                        abstract=result.summary.replace("\n", " ").strip(),
                        published=result.published.isoformat(),
                        updated=result.updated.isoformat(),
                        categories=[c for c in result.categories],
                        primary_category=result.primary_category,
                        pdf_url=result.pdf_url,
                        comment=result.comment,
                        journal_ref=result.journal_ref,
                        doi=result.doi,
                    )

                    seen_ids.add(arxiv_id)
                    self.papers[arxiv_id] = paper
                    logger.info(f"  [{len(seen_ids)}] {arxiv_id}: {paper.title[:80]}")

            except Exception as e:
                logger.error(f"Error searching '{query}': {e}")
                time.sleep(5)
                continue

        logger.info(f"Found {len(self.papers)} unique papers")
        return self.papers

    def download_pdfs(self, batch_size: int = 10, delay: float = 1.0) -> int:
        downloaded = 0
        papers_list = list(self.papers.values())

        for i, paper in enumerate(papers_list):
            pdf_path = self.pdfs_dir / f"{paper.arxiv_id}.pdf"
            if pdf_path.exists():
                paper.pdf_path = str(pdf_path)
                downloaded += 1
                continue

            try:
                url = paper.pdf_url
                if not url.endswith(".pdf"):
                    url = url.replace("/abs/", "/pdf/")
                    if not url.endswith(".pdf"):
                        url += ".pdf"

                with httpx.Client(timeout=60.0, follow_redirects=True) as client:
                    response = client.get(url)
                    response.raise_for_status()

                    with open(pdf_path, "wb") as f:
                        f.write(response.content)

                paper.pdf_path = str(pdf_path)
                downloaded += 1
                logger.info(f"  [{downloaded}/{len(papers_list)}] Downloaded {paper.arxiv_id}")

                if (i + 1) % batch_size == 0:
                    time.sleep(delay * 2)
                else:
                    time.sleep(delay)

            except Exception as e:
                logger.warning(f"  Failed to download {paper.arxiv_id}: {e}")
                continue

        logger.info(f"Downloaded {downloaded}/{len(papers_list)} PDFs")
        return downloaded

    def save_metadata(self) -> None:
        data = {aid: asdict(p) for aid, p in self.papers.items()}
        with open(self.metadata_file, "w") as f:
            json.dump(data, f, indent=2, default=str)
        logger.info(f"Saved metadata for {len(self.papers)} papers to {self.metadata_file}")

    def load_metadata(self) -> dict[str, PaperMetadata]:
        if not self.metadata_file.exists():
            logger.warning(f"No metadata file found at {self.metadata_file}")
            return {}

        with open(self.metadata_file, "r") as f:
            data = json.load(f)

        self.papers = {}
        for aid, pdata in data.items():
            pdata.pop("text_extracted", None)
            pdata.pop("citations_parsed", None)
            pdata.pop("references", None)
            self.papers[aid] = PaperMetadata(
                **{k: v for k, v in pdata.items() if k in PaperMetadata.__dataclass_fields__}
            )

        logger.info(f"Loaded metadata for {len(self.papers)} papers")
        return self.papers

    def collect(self, download: bool = True) -> dict[str, PaperMetadata]:
        logger.info("Starting paper collection...")
        self.search_papers()

        if download:
            logger.info("Downloading PDFs...")
            self.download_pdfs()

        self.save_metadata()
        logger.info(f"Collection complete: {len(self.papers)} papers")
        return self.papers


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(message)s")
    scraper = ArxivScraper(max_papers=500)
    scraper.collect(download=True)


if __name__ == "__main__":
    main()
