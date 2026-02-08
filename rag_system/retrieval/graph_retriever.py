import logging
import random
from typing import Optional

import numpy as np

from rag_system.retrieval.dense_retriever import RetrievalResult

logger = logging.getLogger(__name__)


class GraphRetriever:
    def __init__(self, graph_builder, paper_embeddings: Optional[dict] = None):
        self.graph_builder = graph_builder
        self.paper_embeddings = paper_embeddings or {}

    def citation_expansion(
        self,
        seed_arxiv_ids: list[str],
        max_results: int = 20,
    ) -> list[RetrievalResult]:
        expanded: dict[str, float] = {}

        for seed_id in seed_arxiv_ids:
            cited_by_seed = self.graph_builder.get_papers_cited_by(seed_id)
            for cited_id in cited_by_seed:
                if cited_id not in seed_arxiv_ids:
                    score = expanded.get(cited_id, 0.0)
                    expanded[cited_id] = max(score, 0.8)

            citing_seed = self.graph_builder.get_papers_citing(seed_id)
            for citing_id in citing_seed:
                if citing_id not in seed_arxiv_ids:
                    score = expanded.get(citing_id, 0.0)
                    expanded[citing_id] = max(score, 0.7)

            second_hop_cited = set()
            for cited_id in cited_by_seed[:5]:
                for second_id in self.graph_builder.get_papers_cited_by(cited_id):
                    if second_id not in seed_arxiv_ids and second_id not in expanded:
                        second_hop_cited.add(second_id)

            for second_id in second_hop_cited:
                score = expanded.get(second_id, 0.0)
                expanded[second_id] = max(score, 0.4)

        sorted_papers = sorted(expanded.items(), key=lambda x: x[1], reverse=True)[:max_results]

        results: list[RetrievalResult] = []
        for arxiv_id, score in sorted_papers:
            node_data = self.graph_builder.graph.nodes.get(arxiv_id, {})
            results.append(
                RetrievalResult(
                    chunk_id=f"{arxiv_id}_graph",
                    score=score,
                    text=node_data.get("abstract", ""),
                    arxiv_id=arxiv_id,
                    paper_title=node_data.get("title", ""),
                    section_name="Graph Expansion",
                    metadata={"retrieval_method": "citation_expansion"},
                )
            )

        return results

    def graph_walk(
        self,
        seed_arxiv_ids: list[str],
        query_embedding: Optional[np.ndarray] = None,
        num_walks: int = 10,
        walk_length: int = 5,
        max_results: int = 20,
    ) -> list[RetrievalResult]:
        visit_counts: dict[str, float] = {}

        for _ in range(num_walks):
            current = random.choice(seed_arxiv_ids) if seed_arxiv_ids else None
            if current is None or current not in self.graph_builder.graph:
                continue

            for step in range(walk_length):
                neighbors = list(self.graph_builder.graph.successors(current)) + \
                            list(self.graph_builder.graph.predecessors(current))

                if not neighbors:
                    break

                if query_embedding is not None and self.paper_embeddings:
                    neighbor_scores = []
                    for n in neighbors:
                        if n in self.paper_embeddings:
                            sim = float(np.dot(query_embedding, self.paper_embeddings[n]))
                            neighbor_scores.append((n, max(sim, 0.01)))
                        else:
                            neighbor_scores.append((n, 0.1))

                    total = sum(s for _, s in neighbor_scores)
                    probs = [s / total for _, s in neighbor_scores]
                    candidates = [n for n, _ in neighbor_scores]
                    current = random.choices(candidates, weights=probs, k=1)[0]
                else:
                    current = random.choice(neighbors)

                if current not in seed_arxiv_ids:
                    decay = 1.0 / (step + 1)
                    visit_counts[current] = visit_counts.get(current, 0) + decay

        if visit_counts:
            max_count = max(visit_counts.values())
            for k in visit_counts:
                visit_counts[k] /= max_count

        sorted_papers = sorted(visit_counts.items(), key=lambda x: x[1], reverse=True)[:max_results]

        results: list[RetrievalResult] = []
        for arxiv_id, score in sorted_papers:
            node_data = self.graph_builder.graph.nodes.get(arxiv_id, {})
            results.append(
                RetrievalResult(
                    chunk_id=f"{arxiv_id}_walk",
                    score=score,
                    text=node_data.get("abstract", ""),
                    arxiv_id=arxiv_id,
                    paper_title=node_data.get("title", ""),
                    section_name="Graph Walk",
                    metadata={"retrieval_method": "graph_walk"},
                )
            )

        return results

    def community_retrieval(
        self,
        seed_arxiv_ids: list[str],
        max_results: int = 20,
    ) -> list[RetrievalResult]:
        community_members: dict[str, float] = {}

        for seed_id in seed_arxiv_ids:
            members = self.graph_builder.get_community_members(seed_id)
            pagerank = self.graph_builder.pagerank_scores

            for member_id in members:
                if member_id not in seed_arxiv_ids:
                    pr_score = pagerank.get(member_id, 0.0)
                    current = community_members.get(member_id, 0.0)
                    community_members[member_id] = max(current, pr_score)

        if community_members:
            max_score = max(community_members.values())
            if max_score > 0:
                for k in community_members:
                    community_members[k] /= max_score

        sorted_papers = sorted(
            community_members.items(), key=lambda x: x[1], reverse=True
        )[:max_results]

        results: list[RetrievalResult] = []
        for arxiv_id, score in sorted_papers:
            node_data = self.graph_builder.graph.nodes.get(arxiv_id, {})
            results.append(
                RetrievalResult(
                    chunk_id=f"{arxiv_id}_community",
                    score=score,
                    text=node_data.get("abstract", ""),
                    arxiv_id=arxiv_id,
                    paper_title=node_data.get("title", ""),
                    section_name="Community Retrieval",
                    metadata={"retrieval_method": "community_retrieval"},
                )
            )

        return results

    def get_graph_score(
        self,
        arxiv_id: str,
        seed_arxiv_ids: list[str],
    ) -> dict[str, float]:
        pagerank = self.graph_builder.pagerank_scores.get(arxiv_id, 0.0)

        min_distance = float("inf")
        for seed_id in seed_arxiv_ids:
            dist = self.graph_builder.get_graph_distance(seed_id, arxiv_id)
            if 0 < dist < min_distance:
                min_distance = dist

        if min_distance == float("inf"):
            proximity = 0.0
        else:
            proximity = 1.0 / (1.0 + min_distance)

        return {
            "pagerank": pagerank,
            "graph_proximity": proximity,
        }
