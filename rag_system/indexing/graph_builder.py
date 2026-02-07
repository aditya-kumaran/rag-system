import json
import logging
from pathlib import Path

import networkx as nx

logger = logging.getLogger(__name__)


class CitationGraphBuilder:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.pagerank_scores: dict[str, float] = {}
        self.communities: dict[str, int] = {}

    def build_graph(
        self,
        papers_metadata: dict,
        citation_data: dict,
    ) -> nx.DiGraph:
        for arxiv_id, meta in papers_metadata.items():
            if isinstance(meta, dict):
                title = meta.get("title", "")
                authors = meta.get("authors", [])
                published = meta.get("published", "")
                abstract = meta.get("abstract", "")
                categories = meta.get("categories", [])
            else:
                title = getattr(meta, "title", "")
                authors = getattr(meta, "authors", [])
                published = getattr(meta, "published", "")
                abstract = getattr(meta, "abstract", "")
                categories = getattr(meta, "categories", [])

            self.graph.add_node(
                arxiv_id,
                title=title,
                authors=authors if isinstance(authors, list) else [],
                published=str(published),
                abstract=str(abstract)[:500],
                categories=categories if isinstance(categories, list) else [],
            )

        edge_count = 0
        for source_id, cdata in citation_data.items():
            cited_ids = []
            if isinstance(cdata, dict):
                cited_ids = cdata.get("cited_arxiv_ids", [])
            else:
                cited_ids = getattr(cdata, "cited_arxiv_ids", [])

            for target_id in cited_ids:
                if target_id in self.graph.nodes:
                    self.graph.add_edge(source_id, target_id)
                    edge_count += 1

        logger.info(
            f"Built citation graph: {self.graph.number_of_nodes()} nodes, "
            f"{self.graph.number_of_edges()} edges"
        )
        return self.graph

    def compute_pagerank(self, alpha: float = 0.85) -> dict[str, float]:
        if self.graph.number_of_nodes() == 0:
            return {}

        self.pagerank_scores = nx.pagerank(self.graph, alpha=alpha)
        top_papers = sorted(
            self.pagerank_scores.items(), key=lambda x: x[1], reverse=True
        )[:20]

        logger.info("Top 20 papers by PageRank:")
        for arxiv_id, score in top_papers:
            title = self.graph.nodes[arxiv_id].get("title", "Unknown")
            logger.info(f"  {arxiv_id} ({score:.6f}): {title[:80]}")

        return self.pagerank_scores

    def detect_communities(self) -> dict[str, int]:
        try:
            from community import community_louvain
            undirected = self.graph.to_undirected()
            self.communities = community_louvain.best_partition(undirected)
        except ImportError:
            from networkx.algorithms.community import greedy_modularity_communities
            undirected = self.graph.to_undirected()
            community_sets = greedy_modularity_communities(undirected)
            self.communities = {}
            for comm_id, members in enumerate(community_sets):
                for node in members:
                    self.communities[node] = comm_id

        num_communities = len(set(self.communities.values()))
        logger.info(f"Detected {num_communities} communities")

        comm_sizes: dict[int, int] = {}
        for comm_id in self.communities.values():
            comm_sizes[comm_id] = comm_sizes.get(comm_id, 0) + 1

        for comm_id, size in sorted(comm_sizes.items(), key=lambda x: x[1], reverse=True)[:10]:
            members = [n for n, c in self.communities.items() if c == comm_id]
            titles = [
                self.graph.nodes[n].get("title", "")[:60] for n in members[:3]
            ]
            logger.info(f"  Community {comm_id} ({size} papers): {titles}")

        return self.communities

    def get_citation_count(self, arxiv_id: str) -> int:
        if arxiv_id not in self.graph:
            return 0
        return self.graph.in_degree(arxiv_id)

    def get_papers_cited_by(self, arxiv_id: str) -> list[str]:
        if arxiv_id not in self.graph:
            return []
        return list(self.graph.successors(arxiv_id))

    def get_papers_citing(self, arxiv_id: str) -> list[str]:
        if arxiv_id not in self.graph:
            return []
        return list(self.graph.predecessors(arxiv_id))

    def get_community_members(self, arxiv_id: str) -> list[str]:
        if arxiv_id not in self.communities:
            return []
        comm_id = self.communities[arxiv_id]
        return [n for n, c in self.communities.items() if c == comm_id and n != arxiv_id]

    def get_graph_distance(self, source: str, target: str) -> int:
        if source not in self.graph or target not in self.graph:
            return -1
        try:
            undirected = self.graph.to_undirected()
            return nx.shortest_path_length(undirected, source, target)
        except nx.NetworkXNoPath:
            return -1

    def get_neighbors(self, arxiv_id: str, max_distance: int = 2) -> dict[str, int]:
        if arxiv_id not in self.graph:
            return {}

        undirected = self.graph.to_undirected()
        neighbors: dict[str, int] = {}

        try:
            lengths = nx.single_source_shortest_path_length(
                undirected, arxiv_id, cutoff=max_distance
            )
            for node, dist in lengths.items():
                if node != arxiv_id:
                    neighbors[node] = dist
        except Exception:
            pass

        return neighbors

    def get_graph_stats(self) -> dict:
        if self.graph.number_of_nodes() == 0:
            return {"nodes": 0, "edges": 0}

        in_degrees = [d for _, d in self.graph.in_degree()]
        out_degrees = [d for _, d in self.graph.out_degree()]

        stats = {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "density": nx.density(self.graph),
            "avg_in_degree": sum(in_degrees) / len(in_degrees) if in_degrees else 0,
            "avg_out_degree": sum(out_degrees) / len(out_degrees) if out_degrees else 0,
            "max_in_degree": max(in_degrees) if in_degrees else 0,
            "max_out_degree": max(out_degrees) if out_degrees else 0,
            "num_communities": len(set(self.communities.values())) if self.communities else 0,
        }

        try:
            undirected = self.graph.to_undirected()
            components = list(nx.connected_components(undirected))
            stats["connected_components"] = len(components)
            stats["largest_component_size"] = max(len(c) for c in components)
        except Exception:
            stats["connected_components"] = 0

        return stats

    def save(self, output_dir: str | Path) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        graph_data = nx.node_link_data(self.graph)
        with open(output_dir / "citation_graph.json", "w") as f:
            json.dump(graph_data, f, indent=2, default=str)

        if self.pagerank_scores:
            with open(output_dir / "pagerank_scores.json", "w") as f:
                json.dump(self.pagerank_scores, f, indent=2)

        if self.communities:
            with open(output_dir / "communities.json", "w") as f:
                json.dump(self.communities, f, indent=2)

        stats = self.get_graph_stats()
        with open(output_dir / "graph_stats.json", "w") as f:
            json.dump(stats, f, indent=2)

        logger.info(f"Saved graph data to {output_dir}")

    def load(self, input_dir: str | Path) -> None:
        input_dir = Path(input_dir)

        graph_path = input_dir / "citation_graph.json"
        if graph_path.exists():
            with open(graph_path, "r") as f:
                graph_data = json.load(f)
            self.graph = nx.node_link_graph(graph_data, directed=True)
            logger.info(
                f"Loaded graph: {self.graph.number_of_nodes()} nodes, "
                f"{self.graph.number_of_edges()} edges"
            )

        pagerank_path = input_dir / "pagerank_scores.json"
        if pagerank_path.exists():
            with open(pagerank_path, "r") as f:
                self.pagerank_scores = json.load(f)

        communities_path = input_dir / "communities.json"
        if communities_path.exists():
            with open(communities_path, "r") as f:
                self.communities = json.load(f)
