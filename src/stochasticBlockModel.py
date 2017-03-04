#!/usr/bin/env python
#coding:utf-8
"""
  Purpose:  Stochastic block model
  Created:  01/02/2017
"""

from time import clock
import sys
import numpy as np

########################################################################
class SBM:
    """Stochastic block model class"""

    #----------------------------------------------------------------------
    def __init__(self, n_vertices, n_communities, probability_matrix):
        """
        Constructor
        :param n_vertices: int
        :param n_communities: int
        :param probability_matrix: np.array
        """
        self.n_vertices = n_vertices  # number of vertices
        self.n_communities = n_communities  # number of communities (C1, ..., Ck)
        if(not (probability_matrix.T == probability_matrix).all()):
            raise ValueError("Probability matrix isn't symmetric")
        self.probability_matrix = probability_matrix # matrix of edge probabilities
        self.community_labels = np.random.randint(low=0, high=self.n_communities, size=self.n_vertices, dtype=np.uint8) # community label assigned to each vertex
        self.n_per_community = [len(np.argwhere(self.community_labels == i)) for i in xrange(self.n_communities)] # number of vertices per community (n1, n2, ..., nk)
        self.adjacency_matrix = self.GenerateAjacencyMatrix() # generate adjacency matrix from the Stochastic block model
        self.expected_degrees = np.dot(self.probability_matrix, self.n_per_community) # degrees expectations [E(d1), E(d2), ..., E(dk)]
        self.degrees_vector = np.sum(self.adjacency_matrix, axis=0) # d[i] is the degree of each node i
        self.average_degree = np.mean(self.degrees_vector)

    #----------------------------------------------------------------------
    def GenerateAjacencyMatrix(self):
        """
        Generate adajcency matrix
        :return: np.array
        """
        t0 = clock()
        graph_matrix = np.zeros((self.n_vertices, self.n_vertices), dtype=bool) # adjacency matrix initialization
        for i in xrange(self.n_vertices):
            for j in xrange(i):
                val = self.probability_matrix[self.community_labels[i],self.community_labels[j]]
                p = np.random.rand()
                if p <= val:
                    graph_matrix[i][j] = 1

        graph_matrix += graph_matrix.T # symmetric as graph is undirected
        #print("Time taken to generate the graph: " + str((clock() - t0)*1000) + "ms")
        #print("Graph size in memory: " + str(sys.getsizeof(graph_matrix)/1024) + "ko")
        return graph_matrix
