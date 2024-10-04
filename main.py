"""[Fall 2024] AI607 Graph Mining and Social Network Analysis.
Homework #2 : Citation Network Analysis using PageRank and HITS

This program prints the top 10 PageRank, Hubs, and Authority nodes for the given graph

Usage:
    python main.py -f [file]
"""

from __future__ import absolute_import
import argparse
from graph import CitationNetwork, pagerank, hits, print_top_k

if __name__ == "__main__":
    # Parse the arguments
    parser = argparse.ArgumentParser(
        description="Get the top 10 PageRank, Hubs, and Authority nodes for the given graph"
    )
    parser.add_argument(
        "-f",
        "--file",
        action="store",
        default="graph.txt",
        type=str,
        help="A file path for an initial matrix",
    )
    args = parser.parse_args()

    # Hyperparameter settings
    MAX_ITER = 100  # The number of maximum iterations for PageRank and HITS algorithms
    TOL = 1e-5  # Tolerance for PageRank and HITS algorithms
    K = 10  # To find top-k scores

    # Generate the graph
    graph = CitationNetwork(args.file)

    # Run PageRank algorithm
    for damping_factor in [0.8, 0.85, 0.9]:
        page_ranks = pagerank(graph, damping_factor, MAX_ITER, TOL)
        
        # Print top-k pagerank scores with varying damping factors
        print(f"--- Top-{K} pageranks when damping factor is {damping_factor} ---")
        print_top_k(page_ranks, graph.idx2title, K)
    
    # Run HITS algorithm
    hubs, authorities = hits(graph, MAX_ITER, TOL)

    # Print top-k hub, and authority scores
    print(f"--- Top-{K} hubs ---")
    print_top_k(hubs, graph.idx2title, K)

    print(f"--- Top-{K} authorities ---")
    print_top_k(authorities, graph.idx2title, K)
