#!/usr/bin/env python
#coding:utf-8
"""
  Purpose:  Main file
  Created:  12/02/2017
"""

import networkx as nx
import matplotlib.pyplot as plt
from matrices import *
import spectralClustering
from stochasticBlockModel import SBM
import numpy as np

#----------------------------------------------------------------------
def main():

	#----------------------------------------------------------------------
	# Stochastic block model parameters
	#----------------------------------------------------------------------
	n_vertices = 1000  # number of vertices
	n_communities = 2  # number of communities

	# Fixing cin > cout is referred to as the assortative case, because vertices
	# from the same group connect with higher probability than with vertices from
	# other groups. cout > cin is called the disassortative case. An important conjecture
	# is that any tractable algorithm will only detect communities if
	# abs(cin - cout) > n_communities*sqrt(c), where c is the average degree.
	cin = 15
	cout = 1
	probability_matrix = (1.0/n_vertices)*(np.full((n_communities,n_communities), cout, dtype=int) + np.diag([cin-cout]*n_communities)) # matrix of edge probabilities
	sbm = SBM(n_vertices, n_communities, probability_matrix)
	print("Average degree: " + str(sbm.average_degree))

	#----------------------------------------------------------------------
	# Draw generated graph and print communities
	#----------------------------------------------------------------------
	color_map = np.array(['cyan', 'red', 'yellow', 'magenta', 'blue', 'green', 'white'][:n_communities])
	for i in xrange(n_communities):
		indices = [j+1 for j, x in enumerate(sbm.community_labels) if x == i]
		print("Community C{}, n{} = {} vertices, color: {}, E[di] = {}".format(str(i), i, sbm.n_per_community[i], color_map[i], sbm.expected_degrees[i]))

	if n_vertices > 100:
		print("Can't draw if number of vertices is too big")
	else:
		G = nx.from_numpy_matrix(sbm.adjacency_matrix) # generate networkx graph
		labels = {key: key+1 for key in xrange(n_vertices)} # vertices numbers
		node_color = color_map[sbm.community_labels]
		nx.draw(G, labels=labels, node_color=node_color, font_size=12)
		plt.show()

	eigvals, eigvects = np.linalg.eig(BetheHessian(sbm.adjacency_matrix))
	plt.hist(eigvals, bins=100)
	plt.show()

	##----------------------------------------------------------------------
	## Comparison between sparse and dense graphs
	##----------------------------------------------------------------------
	#x = range(25,700,25)
	#y_sparse = []
	#y_dense = []
	#for i in x:
		#n_vertices = i  # number of vertices
		#n_communities = 2  # number of communities
		#cin = 13
		#cout = 3
		#probability_matrix = (1.0/n_vertices)*(np.full((n_communities,n_communities), cout, dtype=int) + np.diag([cin-cout]*n_communities))
		#sbm = SBM(n_vertices, n_communities, probability_matrix)
		#y_sparse.append(sbm.average_degree)
	#for i in x:
		#n_vertices = i  # number of vertices
		#n_communities = 2  # number of communities
		#cin = 0.7
		#cout = 0.3
		#probability_matrix = np.full((n_communities,n_communities), cout, dtype=int) + np.diag([cin-cout]*n_communities)
		#sbm = SBM(n_vertices, n_communities, probability_matrix)
		#y_dense.append(sbm.average_degree)
	#plt.xlim(0,x[-1])
	#plt.xlabel("Number of vertices")
	#plt.ylabel("Average degree")
	#plt.plot(x,y_sparse, 'r', label="sparse")
	#plt.plot(x,y_dense, 'b', label="dense")
	#plt.legend(loc='upper left')
	#plt.show()




if __name__ == '__main__':
	main()
