#!/usr/bin/env python
#coding:utf-8
"""
  Purpose:  Stochastic block model
  Created:  01/02/2017
"""
import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# Stochastic block model parameters
#----------------------------------------------------------------------
n_vertices = 15  # number of vertices
n_communities = 4  # number of communities
probability_matrix = np.array([
    [0.7, .1, .1, .1],
    [.1, 0.7, .1, .1],
    [.1, .1, 0.7, .1],
    [.1, .1, .1, 0.7],
]) # symetric matrix of edge probabilities
community_labels = [random.randint(0, n_communities-1) for i in xrange(n_vertices)] # community label assigned to each vertices
graph_matrix = np.zeros((n_vertices, n_vertices), dtype=int) # adjacency matrix initialization

# Adjacency matrix generation
#----------------------------------------------------------------------
for i in xrange(n_vertices):
    for j in xrange(i):
        community_i = community_labels[i]
        community_j = community_labels[j]
        val = probability_matrix[community_i][community_j]
        p = random.random()
        if p <= val:
            graph_matrix[i][j] = 1

print(graph_matrix) # print adjacency matrix
G = nx.from_numpy_matrix(graph_matrix) # generate networkx graph

# Draw generated graph and print communities
#----------------------------------------------------------------------
color_map = ['cyan', 'red', 'yellow', 'magenta', 'blue', 'green', 'white'][:n_communities]
node_color = [color_map[i] for i in community_labels]
labels = {key: key+1 for key in xrange(n_vertices)}
nx.draw(G, labels=labels, node_color=node_color, font_size=12)

for i in xrange(n_communities):
    indices = [j+1 for j, x in enumerate(community_labels) if x == i]
    print("Community C{} vertices (color: {}): {}".format(str(i), color_map[i], str(indices)))

plt.show()
