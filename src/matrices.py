#!/usr/bin/env python
# coding:utf-8
"""
  Purpose:  Other tests
  Created:  05/02/2017
"""

import scipy.linalg as linalg
import numpy as np

# ----------------------------------------------------------------------
def LaplacianMatrix(matrix):
    """
    Return the Laplacian Matrix of a given graph
    :param matrix: np.array symetric square matrix
    :return: np.array (Laplacian matrix)
    """
    d = np.sum(matrix, axis=0)
    return (np.diag(d) - matrix)

# ----------------------------------------------------------------------
def LaplacianMatrix_1(matrix):
    """
    Return the symmetric normalized Laplacian matrix of a given graph
    :param matrix: np.array symetric square matrix
    :return: np.array square matrix
    """
    d = np.sum(matrix, axis=0)
    D_inv_square_root = linalg.fractional_matrix_power(np.diag(d), -0.5)
    return np.identity(len(d)) - np.dot(np.dot(D_inv_square_root, matrix), D_inv_square_root)

# ----------------------------------------------------------------------
def LaplacianMatrix_2(matrix):
    """
    Return the random-walk normalized Laplacian matrix of a given graph
    :param matrix: np.array symetric square matrix
    :return: np.array (Laplacian matrix)
    """
    d = np.sum(matrix, axis=0)
    return np.identity(len(d)) - np.linalg.inv(np.diag(d))*matrix

# ----------------------------------------------------------------------
def ModularityMatrix(matrix):
    """
    Return the modularity matrix of a given graph
    :param matrix: np.array symetric square matrix
    :return: np.array square matrix
    """
    d = np.sum(matrix, axis=0)
    return matrix - (np.dot(d, d.T) / np.sum(d))

# ----------------------------------------------------------------------
def BetheHessian(matrix, r=None):
    """
    Return the Bethe Hessian matrix of a given graph.
    By default r = sqrt(sum(di)/n)
    :param matrix: np.array symetric square matrix
    :param r: float (parameter)
    :return: np.array square matrix
    """
    d = np.sum(matrix, axis=0)
    if r is None:
        r = np.sqrt(np.mean(d))
    return (r**2 - 1)*np.identity(len(d)) - r*matrix + np.diag(d)
