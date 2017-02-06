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
        """
        Constructor
        :param n_vertices: int
        :param n_communities: int
        :param probability_matrix: np.array
        """
        self.n_vertices = n_vertices  # number of vertices
        self.n_communities = n_communities  # number of communities
        if(not (probability_matrix.T == probability_matrix).all()):
            raise ValueError("Probability matrix isn't symmetric")
        self.probability_matrix = probability_matrix # matrix of edge probabilities
        self.community_labels = np.random.randint(low=0, high=self.n_communities, size=self.n_vertices, dtype=np.uint8) # community label assigned to each vertex
        self.n_per_community = [len(np.argwhere(self.community_labels == i)) for i in xrange(self.n_communities)] # number of vertices per community (n1, n2, ..., nk)
        self.adjacency_matrix = self.ajacencyMatrix()
        self.expected_degrees = self.degreesExpectations() # expectations [E(d1), E(d2), ..., E(dk)]
        self.degrees_vector = self.degreesVector() # d[i] is the degree of each node i
        self.average_degree = np.mean(self.degrees_vector)

    #----------------------------------------------------------------------
    def ajacencyMatrix(self):
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
        print("Time taken to generate the graph: " + str((clock() - t0)*1000) + "ms")
        print("Graph size in memory: " + str(sys.getsizeof(graph_matrix)/1024) + "ko")
        return graph_matrix

    #----------------------------------------------------------------------
    def degreesExpectations(self):
        '''
        Calculate expected degrees of vertices in each community (C1, ..., Ck)
        :return: np.array
        '''
        return np.dot(self.probability_matrix, self.n_per_community) # return [E(d1), E(d2), ..., E(dk)]

    #----------------------------------------------------------------------
    def degreesVector(self):
        """
        Return vector where d[i] is the degree of each node i
        :return: np.array
        """
        return np.sum(self.adjacency_matrix, axis=0)



if __name__ == '__main__':

    #----------------------------------------------------------------------
    # Stochastic block model parameters
    n_vertices = 1000  # number of vertices
    n_communities = 2  # number of communities

    # Fixing cin > cout is referred to as the assortative case, because vertices
    # from the same group connect with higher probability than with vertices from
    # other groups. cout > cin is called the disassortative case. An important conjecture
    # is that any tractable algorithm will only detect communities if
    # abs(cin - cout) > n_communities*sqrt(c), where c is the average degree.
    cin = 13
    cout = 1
    probability_matrix = (1.0/n_vertices)*(np.full((n_communities,n_communities), cout, dtype=int) + np.diag([cin-cout]*n_communities)) # matrix of edge probabilities
    sbm = SBM(n_vertices, n_communities, probability_matrix)
    print("Average degree: " + str(sbm.average_degree))

    #----------------------------------------------------------------------
    # Draw generated graph and print communities
    color_map = np.array(['cyan', 'red', 'yellow', 'magenta', 'blue', 'green', 'white'][:n_communities])
    for i in xrange(n_communities):
        indices = [j+1 for j, x in enumerate(sbm.community_labels) if x == i]
        print("Community C{}, n{} = {} vertices, color: {}, E[di] = {}".format(str(i), i, sbm.n_per_community[i], color_map[i], sbm.expected_degrees[i]))

    if n_vertices > 100:
        sys.exit(0) # Can't draw if number of vertices is too big
    G = nx.from_numpy_matrix(sbm.adjacency_matrix) # generate networkx graph
    labels = {key: key+1 for key in xrange(n_vertices)} # vertices numbers
    node_color = color_map[sbm.community_labels]
    nx.draw(G, labels=labels, node_color=node_color, font_size=12)
    plt.show()
