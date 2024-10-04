"""This file contains an incomplete implementation of the CitationNetwork class, PageRank algorithm, and HITS algorithm.
Your tasks are as follows:
    1. Complete the CitationNetwork class
    2. Complete the pagerank method
    3. Complete the hits method
    4. Complete the print_top_k method
"""

from __future__ import absolute_import
from typing import Dict, Tuple

############################################################################
# You may import additional python standard libraries, numpy and scipy.
# Other libraries are not allowed.
############################################################################


class CitationNetwork:
    """Graph structure for the analysis of the citation network
    """

    def __init__(self, file_path: str) -> None:
        """The constructor of the CitationNetwork class.
        It parses the input file and generates a graph.

        Args:
            file_path (str): The path of the input file which contains papers and citations
        """

        ######### Task 1. Complete the constructor of CitationNetwork ##########
        # Load the input file and process it to a graph
        # You may declare any class variable or method if needed
        ########################################################################


    ############################################################################
    # You may add additional functions for convenience                         #
    ############################################################################


def pagerank(
    graph: CitationNetwork, beta: float, max_iter: int, tol: float
) -> Dict[int, float]:
    """
    An implementation of the PageRank algorithm.
    It uses the power iteration method to compute PageRank scores.
    It returns the PageRank score of each node.

    Args:
        graph (CitationNetwork): A CitationNetwork
        beta (float): Damping factor
        max_iter (int): Maximum number of iterations in the power iteration method
        tol (float): Error tolerance to check convergence in the power iteration method

    Returns:
        dict_rank (Dict[int, float]): A dictionary where the key is the paper index (int) and
            the value is its PageRank score (float).
    """
    
    ################# Task2. Complete the pagerank function ########################
    # Compute PageRank scores of each node using the power iteration method
    ############################################################################


def hits(
    graph: CitationNetwork, max_iter: int, tol: float
) -> Tuple[Dict[int, float], Dict[int, float]]:
    """An implementation of HITS algorithm.
    It uses the power iteration method to compute hub and authority scores.
    It returns the hub and authority scores of each node.

    Args:
        graph (CitationNetwork): A CitationNetwork
        max_iter (int): Maximum number of iterations in the power iteration method
        tol (float): Error tolerance to check convergence in the power iteration method

    Returns:
        (hubs, authorities) (Tuple[Dict[int, float], Dict[int, float]]): Two-tuple of dictionaries.
            For each dictionary, the key is the paper index (int) and the value is its score (float)
    """

    ################# Task3. Complete the hits function ########################
    # Compute hub and authority scores of each node using the power iteration method
    ############################################################################



def print_top_k(scores: Dict[int, float], titles: Dict[int, str],  k: int) -> None:
    """Print top-k scores in the decreasing order and the corresponding titles and indices.
    The printing format should be as follows:
        <Index 1>\t<top score 1>\t<Title 1>
        <Index 2>\t<top score 2>\t<Title 2>
        ...
        <Index k>\t<top score k>\t<Title k>

    Args:
        scores (Dict[int, float]): PageRank or Hub or Authority scores.
            For each dictionary, the key is the paper index (int) and the value is its score (float)
        titles (Dict[int, str]): A dictionary where the key is the paper index (int) and
            the value is the paper title (str).
        k (int): The number of top scores to print.
    """

    ############## Task4. Complete the print_top_k function ####################
    # Print top-k scores in the decreasing order
    ############################################################################

