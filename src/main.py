#!/usr/bin/env python
#coding:utf-8
"""
  Purpose:  Main file
  Created:  12/02/2017
"""

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from spectralClustering import SpectralClustering
from matrices import *
from stochasticBlockModel import SBM
from testMethode import *
from test import *

PLOT_MAX_NODES = 100

#----------------------------------------------------------------------
def main():
    #----------------------------------------------------------------------
    # Stochastic block model parameters
    #----------------------------------------------------------------------
    n_vertices = 100  # number of vertices
    n_communities = 5  # number of communities

    # Fixing cin > cout is referred to as the assortative case, because vertices
    # from the same group connect with higher probability than with vertices from
    # other groups. cout > cin is called the disassortative case. An important conjecture
    # is that any tractable algorithm will only detect communities if
    # abs(cin - cout) > n_communities*sqrt(c), where c is the average degree.
    cin = 15
    cout = 6
    probability_matrix = (1.0/n_vertices)*(np.full((n_communities,n_communities), cout, dtype=int) + np.diag([cin-cout]*n_communities)) # matrix of edge probabilities
    sbm = SBM(n_vertices, n_communities, probability_matrix)
    print("Average degree: " + str(sbm.average_degree))

    #----------------------------------------------------------------------
    # Draw generated graph and print communities
    #----------------------------------------------------------------------
    color_map = np.array(['cyan', 'red', 'yellow', 'magenta', 'blue', 'green', 'white'][:n_communities])
    for i in xrange(n_communities):
        indices = [j+1 for j, x in enumerate(sbm.community_labels) if x == i]
        print("Community C{}, n{} = {} vertices, color: {}, E[di] = {}".format(str(i), i, sbm.n_per_community[i], color_map[i], sbm.expected_degrees[i]))

    if n_vertices > PLOT_MAX_NODES: print("Can't draw if number of vertices is too big")
    else:
        G = nx.from_numpy_matrix(sbm.adjacency_matrix) # generate networkx graph
        labels = {key: key+1 for key in xrange(n_vertices)} # vertices numbers
        node_color = color_map[sbm.community_labels]
        plt.title("Generated graph using Stochastic block model\n{} nodes and {} communities".format(n_vertices, n_communities))
        nx.draw(G, labels=labels, node_color=node_color, font_size=10)
        plt.figure()

    #----------------------------------------------------------------------
    # Spectral clustering
    #----------------------------------------------------------------------
    n_clusters = 5
    spectral_labels, eigvals, eigvects, W = SpectralClustering(n_clusters, BetheHessian(sbm.adjacency_matrix)) # spectral clustering
    test_methode(sbm.community_labels,spectral_labels,n_communities, n_vertices)

    # Eigenvalues and eigenvectors
    plt.title("Histogram of Bethe Hessian matrix eigenvalues")
    plt.hist(eigvals, bins=100) # plot histogram of the eigenvalues
    plt.figure()
    plt.title("Eigenvectors corresponding to the two smallest eigenvalues")
    plt.plot(W[:,0], W[:,1], 'o', markersize=5) # plot eigenvectors corresponding to the two smallest eigenvalues

    # Kmeans
    plt.figure()
    plt.title("Kmeans")
    for i in xrange(n_clusters):
        ds = W[np.where(spectral_labels == i)]
        plt.plot(ds[:,0], ds[:,1], color=color_map[i], marker='o', markersize=5, ls='')

    if n_vertices > PLOT_MAX_NODES: print("Can't draw if number of vertices is too big")
    else:
        plt.figure()
        plt.title("Detected communities with the Bethe Hessian matrix")
        nx.draw(G, labels=labels, node_color=color_map[spectral_labels], font_size=10)
    plt.show()



if __name__ == '__main__':
    main()
