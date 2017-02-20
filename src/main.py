#!/usr/bin/env python
#coding:utf-8
"""
  Purpose:  Main file
  Created:  12/02/2017
"""

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import spectralClustering
from matrices import *
from stochasticBlockModel import SBM
from sklearn.cluster import KMeans

#----------------------------------------------------------------------
def main():
	#----------------------------------------------------------------------
	# Stochastic block model parameters
	#----------------------------------------------------------------------
	n_vertices = 30  # number of vertices
	n_communities = 2  # number of communities

	# Fixing cin > cout is referred to as the assortative case, because vertices
	# from the same group connect with higher probability than with vertices from
	# other groups. cout > cin is called the disassortative case. An important conjecture
	# is that any tractable algorithm will only detect communities if
	# abs(cin - cout) > n_communities*sqrt(c), where c is the average degree.
	cin = 15
	cout = 6
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

	if n_vertices > 300: print("Can't draw if number of vertices is too big")
	else:
		G = nx.from_numpy_matrix(sbm.adjacency_matrix) # generate networkx graph
		labels = {key: key+1 for key in xrange(n_vertices)} # vertices numbers
		node_color = color_map[sbm.community_labels]
		plt.title("Generated graph using Stochastic block model\n{} nodes and {} communities".format(n_vertices, n_communities))
		nx.draw(G, labels=labels, node_color=node_color, font_size=10)
		plt.figure()

	#----------------------------------------------------------------------
	# Spectral clustering
	#----------------------------------------------------------------------
	n_clusters = 2
	eigvals, eigvects = np.linalg.eig(ModularityMatrix(sbm.adjacency_matrix)) # eigvects[:,i] is the eigenvector corresponding to the eigenvalue eigvals[i]
	plt.title("Histogram of Bethe Hessian matrix eigenvalues")
	plt.hist(eigvals, bins=100) # plot histogram

	indices = eigvals.argsort()[:n_clusters] # find the two smallest eigenvalues indices
	W = np.column_stack((eigvects[:,indices[0]], eigvects[:,indices[1]]))
	plt.figure()
	plt.title("Eigenvectors corresponding to the two smallest eigenvalues")
	plt.plot(W[:,0], W[:,1], 'o', markersize=5)

	kmeans = KMeans(n_clusters=n_clusters).fit(W) # kmeans
	plt.figure()
	plt.title("K-means")
	for i in xrange(n_clusters):
		ds = W[np.where(kmeans.labels_ == i)]
		plt.plot(ds[:,0], ds[:,1], color=color_map[i], marker='o', markersize=5, ls='')

	if n_vertices > 300: print("Can't draw if number of vertices is too big")
	else:
		plt.figure()
		plt.title("Detected communities with the Bethe Hessian matrix")
		nx.draw(G, labels=labels, node_color=color_map[~kmeans.labels_], font_size=10)
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
