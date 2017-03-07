#!/usr/bin/env python
#coding:utf-8
'''
Test to use benchmark graphs
Created 06/03/2017
'''

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matrices import *
from spectralClustering import SpectralClustering
import pylab

graph = nx.read_pajek("../benchmark_graphs/USAir97(6clusters).net", encoding='UTF-8')

n_clusters = 6

matrix = nx.to_numpy_matrix(graph)
matrix = 1*(matrix != 0)
matrix = np.array(matrix)
G = nx.from_numpy_matrix(matrix)
print(matrix)
print(np.size(matrix,0))
print(np.size(matrix,1))
spectral_labels, eigvals, eigvects, W = SpectralClustering(n_clusters, BetheHessian(matrix), "BetheHessian")
print('pop')
n_vertices = np.size(matrix, 0)
color_map = np.array(['cyan', 'red', 'yellow', 'magenta', 'blue', 'green', 'white'])
labels = {key: key+1 for key in xrange(n_vertices)} # vertices numbers
#plt.figure()
#plt.title("Detected communities")
#nx.draw(G, labels=labels, node_color=color_map[spectral_labels], font_size=10)
#pylab.show()
print([(spectral_labels == i).sum() for i in range(0,n_clusters)])