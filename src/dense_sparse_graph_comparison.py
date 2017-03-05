#!/usr/bin/env python
#coding:utf-8
"""
  Purpose:  Comparison between sparse and dense graphs
  Created:  21/02/2017
"""

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from stochasticBlockModel import SBM

x = range(5,150,25)
y_sparse = []
y_dense = []

for i in x: # sparse graph
	n_vertices = i  # number of vertices
	n_communities = 2  # number of communities
	cin = 15
	cout = 5
	probability_matrix = (1.0/n_vertices)*(np.full((n_communities,n_communities), cout) + np.diag([cin-cout]*n_communities))
	sbm = SBM(n_vertices, n_communities, probability_matrix)
	y_sparse.append(sbm.average_degree)

for i in x: # dense graph
	n_vertices = i  # number of vertices
	n_communities = 2  # number of communities
	cin = 0.7
	cout = 0.3
	probability_matrix = np.full((n_communities,n_communities), cout) + np.diag([cin-cout]*n_communities)
	sbm = SBM(n_vertices, n_communities, probability_matrix)
	y_dense.append(sbm.average_degree)

plt.xlim(0,x[-1])
plt.xlabel("Number of vertices")
plt.ylabel("Average degree")
plt.plot(x,y_sparse, 'r', label="sparse")
plt.plot(x,y_dense, 'b', label="dense")
plt.legend(loc='upper left')
plt.show()
