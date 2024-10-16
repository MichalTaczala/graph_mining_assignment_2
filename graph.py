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
import numpy as np
import scipy.sparse as sp

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
        self.nodes = []
        self.graph = None
        self.idx2title = {}
        self._read_file(file_path)


    ############################################################################
    # You may add additional functions for convenience                         #
    ############################################################################

    def _read_file(self, file_path):
        with open(file_path, "r") as file:
            self.nodes = []
            self.nr_of_nodes = int(file.readline())
            self.graph = sp.lil_matrix((self.nr_of_nodes, self.nr_of_nodes), dtype=np.int8)
            title_temp = ""
            for line in file:
                line = line.strip()
                if line.startswith("#*"):
                    title = line[2:]
                    title_temp = title
                elif line.startswith("#index"):
                    index = int(line[6:])
                    self.idx2title[index] = title_temp
                elif line.startswith("#%"):
                    self.graph[index, int(line[2:])] = 1

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

    r_vector_horizontal = np.ones(graph.nr_of_nodes).reshape(1,-1) / graph.nr_of_nodes
    r_vector_vertical = r_vector_horizontal.T
    # r_vector_vertical_old = np.zeros((graph.nr_of_nodes, 1))
    graph.graph_t = graph.graph.T

    sum_of_columns = np.sum(graph.graph_t, axis=0)


    
    m_matrix_float = graph.graph.astype(np.float32)
    sum_of_columns_filled = np.where(sum_of_columns == 0, 1, sum_of_columns)
    m_matrix_t = m_matrix_float / sum_of_columns_filled.T
    m_matrix = m_matrix_t.T
    
    
    
    places_where_sum_of_columns_is_zero = sum_of_columns == 0


    iterations = 0
    l2_norm_value = np.inf


    while iterations < max_iter and l2_norm_value >= tol:


        dangling_nodes_sum = np.sum(r_vector_vertical[places_where_sum_of_columns_is_zero.T], axis=0)

        r_vector_vertical_old = r_vector_vertical
        r_vector_vertical = beta * m_matrix @  r_vector_vertical + (1-beta)/graph.nr_of_nodes+ beta/graph.nr_of_nodes*dangling_nodes_sum
        l2_norm_value = np.linalg.norm(r_vector_vertical - r_vector_vertical_old)
        
        iterations += 1



    dict_rank = {i: r_vector_vertical[i][0] for i in range(graph.nr_of_nodes)}

    return dict_rank

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
    iterations = 0
    l2_norm_value = np.inf
    
    hubs = np.ones(graph.nr_of_nodes).reshape(-1,1)/np.sqrt(graph.nr_of_nodes)
    authorities = np.ones(graph.nr_of_nodes).reshape(-1,1)/np.sqrt(graph.nr_of_nodes)

    while iterations < max_iter and l2_norm_value >= tol:

        hubs_prev = hubs


        hubs_not_normalized = graph.graph@authorities
        hubs = hubs_not_normalized / np.linalg.norm(hubs_not_normalized)

        authorities_not_normalized = graph.graph.T@hubs
        authorities = authorities_not_normalized/ np.linalg.norm(authorities_not_normalized)

        l2_norm_value = np.linalg.norm(hubs - hubs_prev)
        iterations += 1
    return ({i: hubs[i][0] for i in range(graph.nr_of_nodes)}, {i: authorities[i][0] for i in range(graph.nr_of_nodes)})



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
    ordered = {k:v for k,v in sorted(scores.items(), key= lambda item: item[1], reverse=True)}
    ordered_items = list(ordered.items())
    best_10 = ordered_items[:10]
    zipped = list(zip(*best_10))
    best_indexes, best_scores = zipped[0], zipped[1]
    best_titles = [titles[index] for index in best_indexes]
    for index, score, title in zip(best_indexes, best_scores, best_titles):
        print(f"{index}\t{score}\t{title}")

