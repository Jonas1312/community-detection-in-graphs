#!/usr/bin/env python
# coding:utf-8
"""
  Purpose:  Test different matrices for sparse graphs
  Created:  04/03/2017
"""

import matplotlib.pyplot as plt
from stochasticBlockModel import SBM
from matrices import *
from sklearn.metrics.cluster import normalized_mutual_info_score
from spectralClustering import SpectralClustering

x_axis = range(20, 2220, 200)
yNMI_bethehessian = [0]*len(x_axis)
yNMI_modularity = [0]*len(x_axis)
yNMI_laplacian = [0]*len(x_axis)
yNMI_adjacency = [0]*len(x_axis)

n_communities = 2
cin = 15
cout = 6
n_iterations = 2

for n in xrange(n_iterations):
    print("Iteration: " + str(n))
    for i, n_vertices in enumerate(x_axis):
        print("Number of vertices: " + str(n_vertices))
        probability_matrix = (1.0 / n_vertices) * (np.full((n_communities, n_communities), cout) + np.diag([cin - cout] * n_communities))
        sbm = SBM(n_vertices, n_communities, probability_matrix)

        # BetheHessian matrix
        spectral_labels, _, _, _ = SpectralClustering(n_communities, BetheHessian(sbm.adjacency_matrix), "BetheHessian")
        nmi = normalized_mutual_info_score(sbm.community_labels, spectral_labels)
        yNMI_bethehessian[i] += nmi

        # Modularity matrix
        spectral_labels, _, _, _ = SpectralClustering(n_communities, ModularityMatrix(sbm.adjacency_matrix), "ModularityMatrix")
        nmi = normalized_mutual_info_score(sbm.community_labels, spectral_labels)
        yNMI_modularity[i] += nmi

        # Laplacian matrix
        spectral_labels, _, _, _ = SpectralClustering(n_communities, LaplacianMatrix(sbm.adjacency_matrix), "LaplacianMatrix")
        nmi = normalized_mutual_info_score(sbm.community_labels, spectral_labels)
        yNMI_laplacian[i] += nmi

        # Adjacency matrix
        spectral_labels, _, _, _ = SpectralClustering(n_communities, sbm.adjacency_matrix, "AdjacencyMatrix")
        nmi = normalized_mutual_info_score(sbm.community_labels, spectral_labels)
        yNMI_adjacency[i] += nmi

plt.title("Sparse graph, cin = {}, cout = {}, iterations: {}, communities: {}".format(cin, cout, n_iterations, n_communities))
plt.xlim(0, x_axis[-1])
plt.ylim(0,1)
plt.xlabel("Number of vertices")
plt.ylabel("Normalized Mutual Information Score")
plt.plot(x_axis, [y/n_iterations for y in yNMI_bethehessian], 'r', label="Bethe-Hessian")
plt.plot(x_axis, [y/n_iterations for y in yNMI_modularity], 'b', label="Modularity")
plt.plot(x_axis, [y/n_iterations for y in yNMI_laplacian], 'g', label="Laplacian")
plt.plot(x_axis, [y/n_iterations for y in yNMI_adjacency], 'c', label="Adjacency")
plt.legend(loc='upper right')
plt.show()
