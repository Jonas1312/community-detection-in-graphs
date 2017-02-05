#!/usr/bin/env python
#coding:utf-8
"""
  Purpose:  Other tests
  Created:  05/02/2017
"""

import scipy.linalg as linalg

#----------------------------------------------------------------------
def LaplacianMatrix(adjacency_matrix):
	"""Return Laplacian matrix of a given graph"""
	d = degreesVector(adjacency_matrix)
	D_inv_square_root = linalg.fractional_matrix_power(np.diag(d), -0.5)
	return np.dot(np.dot(D_inv_square_root, adjacency_matrix), D_inv_square_root)

#----------------------------------------------------------------------
def LaplacianMatrix_2(adjacency_matrix):    #other form of Laplacian Matrix found on internet
	d = degreesVector(adjacency_matrix)
	return(np.diag(d)-adjacency_matrix)
	
#----------------------------------------------------------------------
def ModularityMatrix(adjacency_matrix):
	"""Return modularity matrix of a given graph"""
	d = degreesVector(adjacency_matrix)
	return adjacency_matrix - np.dot(d,d.T)/np.sum(d)

#----------------------------------------------------------------------
def BetheHessian(adjacency_matrix, r=None):
	"""Return Bethe Hessian matrix of a given graph. By default r = sqrt(sum(di)/n)"""
	d = degreesVector(adjacency_matrix)
	if r is None:
		r = np.sqrt(np.sum(d)/len(d))
	return (r**2 - 1)*np.identity(len(d)) - r*adjacency_matrix + np.diag(d)
