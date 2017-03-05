#!/usr/bin/env python
#coding:utf-8
"""
  Purpose:  Spectral clustering algorithm
  Created:  12/02/2017
"""

import numpy as np
from sklearn.cluster import KMeans

#----------------------------------------------------------------------
def SpectralClustering(n_clusters, matrix, matrix_name):
    """"""
    eigvals, eigvects = np.linalg.eig(matrix) # eigvects[:,i] is the eigenvector corresponding to the eigenvalue eigvals[i]
    if matrix_name in ["BetheHessian", "LaplacianMatrix"]:
        indices = eigvals.argsort()[:n_clusters] # find the 'n_clusters' isolated eigenvalues
    elif matrix_name in ["ModularityMatrix", "AdjacencyMatrix"]:
        indices = eigvals.argsort()[-n_clusters:] # find the 'n_clusters' isolated eigenvalues
    else:
        raise ValueError("Unknown matrix name")
    W = eigvects[:,indices]
    kmeans = KMeans(n_clusters=n_clusters).fit(W) # kmeans
    return kmeans.labels_, eigvals, eigvects, W
