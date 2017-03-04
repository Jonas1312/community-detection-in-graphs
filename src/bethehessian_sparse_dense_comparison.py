#!/usr/bin/env python
# coding:utf-8
"""
  Purpose:  Test of Bethe-Hessian matrix on sparse and dense graphs
  Created:  04/03/2017
"""

import numpy as np
import matplotlib.pyplot as plt
from stochasticBlockModel import SBM
from matrices import BetheHessian
from sklearn.metrics.cluster import normalized_mutual_info_score
from spectralClustering import SpectralClustering

x = range(25, 700, 25)
y_sparse = []
y_dense = []

for i in x:  # sparse graph
    n_vertices = i  # number of vertices
    n_communities = 2  # number of communities
    cin = 13
    cout = 3
    probability_matrix = (1.0 / n_vertices) * (
    np.full((n_communities, n_communities), cout, dtype=int) + np.diag([cin - cout] * n_communities))
    sbm = SBM(n_vertices, n_communities, probability_matrix)
    spectral_labels, eigvals, eigvects, W = SpectralClustering(n_communities, BetheHessian(sbm.adjacency_matrix),
                                                               "BetheHessian")  # spectral clustering
    nmi = normalized_mutual_info_score(sbm.community_labels, spectral_labels)
    y_sparse.append(nmi)

for i in x:  # dense graph
    n_vertices = i  # number of vertices
    n_communities = 2  # number of communities
    cin = 13
    cout = 3
    probability_matrix = np.full((n_communities, n_communities), cout, dtype=int) + np.diag(
        [cin - cout] * n_communities)
    sbm = SBM(n_vertices, n_communities, probability_matrix)
    spectral_labels, eigvals, eigvects, W = SpectralClustering(n_communities, BetheHessian(sbm.adjacency_matrix),
                                                               "BetheHessian")  # spectral clustering
    nmi = normalized_mutual_info_score(sbm.community_labels, spectral_labels)
    y_dense.append(nmi)


plt.xlim(0, x[-1])
plt.title("Test of Bethe-Hessian matrix with sparse and dense adjacency matrix")
plt.xlabel("Number of vertices")
plt.ylabel("Normalized Mutual Information Score")
plt.plot(x, y_sparse, 'r', label="sparse")
plt.plot(x, y_dense, 'b', label="dense")
plt.legend(loc='upper left')
plt.show()
