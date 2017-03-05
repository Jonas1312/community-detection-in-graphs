#!/usr/bin/env python
#coding:utf-8
"""
  Purpose:  cin - cout spectral clustering tests
  Created:  05/03/2017
"""

import matplotlib.pyplot as plt
from stochasticBlockModel import SBM
from matrices import *
from sklearn.metrics.cluster import normalized_mutual_info_score
from spectralClustering import SpectralClustering

n_vertices = 700
x_step = 0.2
x_axis = [(15,i) for i in np.arange(4,8,x_step)]
yNMI_bethehessian = [0]*len(x_axis)
yNMI_modularity = [0]*len(x_axis)
yNMI_laplacian = [0]*len(x_axis)
yNMI_adjacency = [0]*len(x_axis)

n_communities = 2
n_iterations = 1

for n in xrange(n_iterations):
	print("Iteration {}, {} left".format(n, n_iterations-n))
	for i, (cin, cout) in enumerate(x_axis):
		probability_matrix = (1.0 / n_vertices) * (np.full((n_communities, n_communities), cout, dtype=int) + np.diag([cin - cout] * n_communities))
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

plt.title("Sparse graph, iterations: {}, communities: {}, {} vertices".format(n_iterations, n_communities, n_vertices))
plt.ylim(0,1)
plt.xlabel("cin - cout")
plt.ylabel("Normalized Mutual Information Score")
x_axis = [cin-cout for cin, cout in x_axis]
plt.plot(x_axis, [y/n_iterations for y in yNMI_bethehessian], 'r', label="Bethe-Hessian")
plt.plot(x_axis, [y/n_iterations for y in yNMI_modularity], 'b', label="Modularity")
plt.plot(x_axis, [y/n_iterations for y in yNMI_laplacian], 'g', label="Laplacian")
plt.plot(x_axis, [y/n_iterations for y in yNMI_adjacency], 'c', label="Adjacency")
plt.legend(loc='upper right')
plt.show()
