#!/usr/bin/env python
#coding:utf-8
"""
  Purpose:  Stochastic block model
  Created:  01/02/2017
"""
from time import clock
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import sys

########################################################################
class SBM:
    """Stochastic block model class"""

    #----------------------------------------------------------------------
    def __init__(self, n_vertices, n_communities, probability_matrix):
        """Constructor"""
        self.n_vertices = n_vertices  # number of vertices
        self.n_communities = n_communities  # number of communities
        self.probability_matrix = probability_matrix # matrix of edge probabilities
        self.community_labels = np.random.randint(low=0, high=self.n_communities, size=self.n_vertices) # community label assigned to each vertex
        self.n_per_community = [len(np.argwhere(self.community_labels == i)) for i in xrange(self.n_communities)] # number of vertices per community (n1, n2, ..., nk)
        self.adjacency_matrix = self.ajacencyMatrix()
        self.expected_degrees = self.degreesExpectations() # expectations [E(d1), E(d2), ..., E(dk)]
        self.degrees_vector = self.degreesVector() # d[i] is the degree of each node i
        self.average_degree = np.mean(self.degrees_vector)

    #----------------------------------------------------------------------
    def ajacencyMatrix(self):
        """Generate adajcency matrix"""
        t0 = clock()
        graph_matrix = np.zeros((self.n_vertices, self.n_vertices), dtype=bool) # adjacency matrix initialization
        for i in xrange(self.n_vertices):
            for j in xrange(i):
                val = self.probability_matrix[self.community_labels[i],self.community_labels[j]]
                p = np.random.rand()
                if p <= val:
                    graph_matrix[i][j] = 1

        graph_matrix += graph_matrix.T # symmetric as graph is undirected
        print("Time taken to generate the graph: " + str((clock() - t0)*1000) + "ms")
        print("Graph size in memory: " + str(sys.getsizeof(graph_matrix)/1024) + "ko")
        return graph_matrix

    #----------------------------------------------------------------------
    def degreesExpectations(self):
        """Calculate expected degrees of vertices in each community (C1, ..., Ck)"""
        return np.dot(self.probability_matrix[:len(self.n_per_community),:len(self.n_per_community)], self.n_per_community) # return [E(d1), E(d2), ..., E(dk)]

    #----------------------------------------------------------------------
    def degreesVector(self):
        """Return vector where d[i] is the degree of each node i"""
        return np.sum(self.adjacency_matrix, axis=0)



if __name__ == '__main__':

    #----------------------------------------------------------------------
    # Stochastic block model parameters
    n_vertices = 200  # number of vertices
    n_communities = 3  # number of communities
    probability_matrix = np.array([
        [0.9, .1, .1, .01],
        [.1, 0.9, .9, .01],
        [.1, .9, 0.5, .011],
        [.01, .01, .01, 0.1],
    ]) # matrix of edge probabilities

    sbm = SBM(n_vertices, n_communities, probability_matrix)
    print(sbm.average_degree)

    #----------------------------------------------------------------------
    # Draw generated graph and print communities
    labels = {key: key+1 for key in xrange(n_vertices)} # vertices numbers
    color_map = np.array(['cyan', 'red', 'yellow', 'magenta', 'blue', 'green', 'white'][:n_communities])
    node_color = color_map[sbm.community_labels]
    for i in xrange(n_communities):
        indices = [j+1 for j, x in enumerate(sbm.community_labels) if x == i]
        print("Community C{}, color: {}, E[di] = {}".format(str(i), color_map[i], sbm.expected_degrees[i]))

    if n_vertices > 100: sys.exit(0) # Can't draw if number of vertices is too big
    G = nx.from_numpy_matrix(sbm.adjacency_matrix) # generate networkx graph
    nx.draw(G, labels=labels, node_color=node_color, font_size=12)

    plt.show()
