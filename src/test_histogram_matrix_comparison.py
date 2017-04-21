#!/usr/bin/env python
#coding:utf-8
"""
  Purpose:  Test multiple "r" for Bethe Hessian matrix
  Created:  04/03/2017
"""

import numpy as np
import matplotlib.pyplot as plt
from spectralClustering import SpectralClustering
from matrices import *
from stochasticBlockModel import SBM
from sklearn.metrics.cluster import normalized_mutual_info_score

n_vertices = 1500  # number of vertices
n_communities = 2  # number of communities

# cin > cout is referred to as the assortative case
# cout > cin is called the disassortative case
cin = 0.6
cout = 0.4
probability_matrix = (np.full((n_communities,n_communities), cout) + np.diag([cin-cout]*n_communities)) # matrix of edge probabilities (to generate a sparse graph)
sbm = SBM(n_vertices, n_communities, probability_matrix)

best_r = np.sqrt(np.mean(sbm.average_degree))

y_axis = []

plt.figure(1)
plt.suptitle("Histogram of eigenvalues for different choices of r (dense graph)")

spectral_labels, eigvals, eigvects, W = SpectralClustering(n_communities, sbm.adjacency_matrix, "AdjacencyMatrix") # spectral clustering
eigvals = np.sort(eigvals)
plt.subplot(2,2,1)
plt.hist(eigvals, bins=100) # plot histogram of the eigenvalues
plt.plot([eigvals[-1], eigvals[-1]], [0, 20], 'r', linewidth = 0.6)
plt.plot([eigvals[-2], eigvals[-2]], [0, 20], 'r', linewidth = 0.6)
nmi = normalized_mutual_info_score(sbm.community_labels, spectral_labels)
plt.title("Adjacency Matrix (nmi = {})".format(nmi))
y_axis.append(nmi)
print("Normalized Mutual Information for Adjacency Matrix")

spectral_labels, eigvals, eigvects, W = SpectralClustering(n_communities, BetheHessian(sbm.adjacency_matrix), "BetheHessian") # spectral clustering
eigvals = np.sort(eigvals)
plt.subplot(2,2,2)
plt.hist(eigvals, bins=100) # plot histogram of the eigenvalues
plt.plot([eigvals[0], eigvals[0]], [0, 20], 'r', linewidth = 0.6)
plt.plot([eigvals[1], eigvals[1]], [0, 20], 'r', linewidth = 0.6)
nmi = normalized_mutual_info_score(sbm.community_labels, spectral_labels)
plt.title("BetheHessian (nmi = {})".format(nmi))
y_axis.append(nmi)
print("Normalized Mutual Information for Bethe-Hessian Matrix")

spectral_labels, eigvals, eigvects, W = SpectralClustering(n_communities, LaplacianMatrix(sbm.adjacency_matrix), "LaplacianMatrix") # spectral clustering
eigvals = np.sort(eigvals)
plt.subplot(2,2,3)
plt.hist(eigvals, bins=100) # plot histogram of the eigenvalues
plt.plot([eigvals[0], eigvals[0]], [0, 20], 'r', linewidth = 0.6)
plt.plot([eigvals[1], eigvals[1]], [0, 20], 'r', linewidth = 0.6)
nmi = normalized_mutual_info_score(sbm.community_labels, spectral_labels)
plt.title("LaplacianMatrix (nmi = {})".format(nmi))
y_axis.append(nmi)
print("Normalized Mutual Information for Laplacian Matrix")

spectral_labels, eigvals, eigvects, W = SpectralClustering(n_communities, ModularityMatrix(sbm.adjacency_matrix), "ModularityMatrix") # spectral clustering
eigvals = np.sort(eigvals)
plt.subplot(2,2,4)
plt.hist(eigvals, bins=100) # plot histogram of the eigenvalues
plt.plot([eigvals[-1], eigvals[-1]], [0, 20], 'r', linewidth = 0.6)
plt.plot([eigvals[-2], eigvals[-2]], [0, 20], 'r', linewidth = 0.6)
nmi = normalized_mutual_info_score(sbm.community_labels, spectral_labels)
plt.title("Modularity Matrix (nmi = {})".format(nmi))
y_axis.append(nmi)
print("Normalized Mutual Information for Modularity Matrix")

plt.figure(2)
plt.ylim(0,1)
plt.title("NMI = f(r), cin = {}, cout = {}, n_vertices = {}".format(cin, cout, n_vertices))
plt.xlabel("Matrix")
plt.ylabel("NMI")
plt.plot(y_axis)
plt.show()
