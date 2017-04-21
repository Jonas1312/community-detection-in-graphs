#!/usr/bin/env python
#coding:utf-8
"""
  Purpose:  Test multiple "r" for Bethe Hessian matrix
  Created:  04/03/2017
"""

import numpy as np
import matplotlib.pyplot as plt
from spectralClustering import SpectralClustering
from matrices import BetheHessian
from stochasticBlockModel import SBM
from sklearn.metrics.cluster import normalized_mutual_info_score

n_vertices = 1500  # number of vertices
n_communities = 2  # number of communities

# cin > cout is referred to as the assortative case
# cout > cin is called the disassortative case
cin = 7
cout = 1
probability_matrix = (1.0/n_vertices)*(np.full((n_communities,n_communities), cout) + np.diag([cin-cout]*n_communities)) # matrix of edge probabilities (to generate a sparse graph)
sbm = SBM(n_vertices, n_communities, probability_matrix)

best_r = np.sqrt(np.mean(sbm.average_degree))
print("r = sqrt(average degree) =  " + str(best_r))

x_axis = sorted([1.1, 2, 3, 3.5, 4, 5, 6, best_r])
y_axis = []

plt.figure(1)
plt.suptitle("Histogram of eigenvalues for different choices of r")

for i, r in enumerate(x_axis):
	spectral_labels, eigvals, eigvects, W = SpectralClustering(n_communities, BetheHessian(sbm.adjacency_matrix, r), "BetheHessian") # spectral clustering
	eigvals = np.sort(eigvals)
	plt.subplot(2,4,i+1)
	plt.title("r = " + str(r))
	plt.hist(eigvals, bins=100) # plot histogram of the eigenvalues
	plt.plot([eigvals[0], eigvals[0]], [0, 20], 'r', linewidth = 0.6)
	plt.plot([eigvals[1], eigvals[1]], [0, 20], 'r', linewidth = 0.6)
	nmi = normalized_mutual_info_score(sbm.community_labels, spectral_labels)
	if r == best_r: best_nmi = nmi
	y_axis.append(nmi)
	print("Normalized Mutual Information for r = {} : {}".format(r, str(nmi)))

plt.figure(2)
plt.ylim(0,1)
plt.title("NMI = f(r), cin = {}, cout = {}, n_vertices = {}".format(cin, cout, n_vertices))
plt.xlabel("r")
plt.ylabel("NMI")
plt.plot(x_axis, y_axis)
plt.plot(best_r, best_nmi, 'ro', markersize=10)
plt.show()
