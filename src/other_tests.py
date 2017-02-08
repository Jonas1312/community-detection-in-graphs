#!/usr/bin/env python
# coding:utf-8
"""
  Purpose:  Other tests
  Created:  05/02/2017
"""

import scipy.linalg as linalg


# ----------------------------------------------------------------------
def LaplacianMatrix(adjacency_matrix):
    """
    Return Laplacian matrix of a given graph (first version)
    :param adjacency_matrix: np.array symetric square matrix
    :return: np.array square matrix
    """
    d = degreesVector(adjacency_matrix)
    D_inv_square_root = linalg.fractional_matrix_power(np.diag(d), -0.5)
    return np.dot(np.dot(D_inv_square_root, adjacency_matrix), D_inv_square_root)


# ----------------------------------------------------------------------
def LaplacianMatrix_2(adjacency_matrix):
    """
    return an other form of Laplacian Matrix found on internet
    :param adjacency_matrix: np.array symetric square matrix
    :return: np.array (Laplacian matrix)
    """
    d = degreesVector(adjacency_matrix)
    return (np.diag(d) - adjacency_matrix)


# ----------------------------------------------------------------------
def ModularityMatrix(adjacency_matrix):
    """
    Return modularity matrix of a given graph
    :param adjacency_matrix: np.array symetric square matrix
    :return: np.array square matrix
    """
    d = degreesVector(adjacency_matrix)
    return adjacency_matrix - np.dot(d, d.T) / np.sum(d)


# ----------------------------------------------------------------------
def BetheHessian(adjacency_matrix, r=None):
    """
    Return Bethe Hessian matrix of a given graph. By default r = sqrt(sum(di)/n)
    :param adjacency_matrix: np.array symetric square matrix
    :param r: float (parameter)
    :return: np.array square matrix
    """
    d = degreesVector(adjacency_matrix)
    if r is None:
        r = np.sqrt(np.sum(d) / len(d))
    return (r ** 2 - 1) * np.identity(len(d)) - r * adjacency_matrix + np.diag(d)


# -----------------------------------------------------------------------
def Kmean(adjacency_matrix):
    """
    K mean adapted to graphs (2 communities)
    return a list of -1 (fist community) and +1 (second community). 0 = undefined community
    :param adjacency_matrix:
    :return: np.array
    """
    communities = np.zeros(n_vertices)
    communities_temp = np.zeros(n_vertices)
    s0 = int(np.random.rand() * n_vertices)
    communities[s0] = 1
    print('initialisation community 1 ' + str(s0))
    s1 = int(np.random.rand() * n_vertices)
    while (s1 == s0):
        s1 = int(np.random.rand() * n_vertices)
    communities[s1] = -1
    print('initialisation community 2 ' + str(s1))
    n_iterations = 5  # nombre d'itérations
    for iteration in xrange(n_iterations):
        communities_temp = np.zeros(n_vertices)
        for s in xrange(n_vertices):
            comm1 = 0
            comm2 = 0
            for p in xrange(n_vertices):
                if (communities[p] == -1) and adjacency_matrix[p, s] == True:
                    comm1 += 1
                elif (communities[p] == 1) and adjacency_matrix[p, s] == True:
                    comm2 += 1
                if comm1 > comm2:
                    communities_temp[s] = -1
                elif comm1 < comm2:
                    communities_temp[s] = 1
        communities, communities_temp = communities_temp, communities
    return (communities)  # +1 -> community one, -1 -> community 2, 0 : undefined
