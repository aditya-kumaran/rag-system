import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

DATA_DIR = Path("data")


def phase1_collect():
    from rag_system.data_collection.arxiv_scraper import ArxivScraper

    logger.info("=== Phase 1: Data Collection ===")
    scraper = ArxivScraper(data_dir=DATA_DIR, max_papers=500)
    papers = scraper.collect(download=True)
    logger.info(f"Collected {len(papers)} papers")
    return papers


def phase1_extract(papers_metadata: dict):
    from rag_system.data_collection.pdf_extractor import extract_papers_batch

    logger.info("=== Phase 1: PDF Extraction ===")
    pdfs_dir = DATA_DIR / "pdfs"
    output_dir = DATA_DIR / "papers"

    meta_dict = {}
    for aid, meta in papers_metadata.items():
        if hasattr(meta, "__dict__"):
            from dataclasses import asdict
            meta_dict[aid] = asdict(meta)
        else:
            meta_dict[aid] = meta

    extracted = extract_papers_batch(meta_dict, pdfs_dir, output_dir)
    logger.info(f"Extracted text from {len(extracted)} papers")
    return extracted


def phase1_citations(papers_metadata: dict, extracted_papers: dict):
    from rag_system.data_collection.citation_parser import CitationParser

    logger.info("=== Phase 1: Citation Parsing ===")

    known = {}
    for aid, meta in papers_metadata.items():
        if isinstance(meta, dict):
            known[aid] = meta
        else:
            known[aid] = {"title": getattr(meta, "title", ""), "arxiv_id": aid}

    parser = CitationParser(known_papers=known)
    citation_data = parser.parse_all_papers(
        extracted_papers,
        output_path=DATA_DIR / "citation_data.json",
    )
    logger.info(f"Parsed citations for {len(citation_data)} papers")
    return citation_data


def phase2_build_index(papers_metadata: dict, extracted_papers: dict, citation_data: dict):
    from rag_system.pipeline import RAGPipeline

    logger.info("=== Phase 2-3: Building Index ===")
    pipeline = RAGPipeline(data_dir=DATA_DIR)
    pipeline.build_index(papers_metadata, extracted_papers, citation_data)
    logger.info("Index built successfully")
    return pipeline


def phase6_evaluate(pipeline=None):
    from rag_system.evaluation.evaluator import Evaluator
    from rag_system.pipeline import RAGPipeline

    logger.info("=== Phase 6: Evaluation ===")

    if pipeline is None:
        pipeline = RAGPipeline.load_from_disk(data_dir=DATA_DIR)

    if pipeline is None:
        logger.error("Pipeline not initialized. Run build_index first.")
        return

    evaluator = Evaluator(
        hybrid_retriever=pipeline.hybrid_retriever,
        reranker=pipeline.reranker,
        llm_client=pipeline.llm_client,
    )

    logger.info("Running ablation study...")
    ablation_results = evaluator.run_ablation_study()

    logger.info("Running generation evaluation...")
    generation_results = evaluator.evaluate_generation(num_questions=20)

    report = evaluator.generate_report(
        ablation_results,
        generation_results,
        output_path=DATA_DIR / "evaluation_report.md",
    )
    logger.info(f"\n{report}")

    return ablation_results, generation_results


def run_all():
    logger.info("Starting full pipeline...")

    papers = phase1_collect()
    extracted = phase1_extract(papers)
    citations = phase1_citations(papers, extracted)
    pipeline = phase2_build_index(papers, extracted, citations)
    phase6_evaluate(pipeline)

    logger.info("Full pipeline complete!")


def run_api():
    import uvicorn
    uvicorn.run("rag_system.api.main:app", host="0.0.0.0", port=8000, reload=True)


def run_ui():
    from rag_system.ui.app import main
    main()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        command = sys.argv[1]
        if command == "collect":
            phase1_collect()
        elif command == "api":
            run_api()
        elif command == "ui":
            run_ui()
        elif command == "evaluate":
            phase6_evaluate()
        elif command == "all":
            run_all()
        else:
            print(f"Unknown command: {command}")
            print("Usage: python -m rag_system.run_pipeline [collect|api|ui|evaluate|all]")
    else:
        run_all()
