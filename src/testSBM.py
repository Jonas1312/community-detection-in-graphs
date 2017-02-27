#!/usr/bin/env python
# coding:utf-8

from stochasticBlockModel import SBM
import matplotlib.pyplot as plt
from time import clock
import numpy as np

cin = 15
cout = 5
n_communities = 2  # number of communities


n_vertices_test = [100 * n for n in xrange(1,26)]# number of vertices
times_test = []

for n_vertices in n_vertices_test:

    probability_matrix = (1.0 / n_vertices) * (np.full((n_communities, n_communities), cout, dtype=int) + np.diag(
        [cin - cout] * n_communities))  # matrix of edge probabilities
    time1 = clock()
    sbm = SBM(n_vertices, n_communities, probability_matrix)
    time2 = clock()
    times_test.append(time2-time1)
plt.plot(n_vertices_test,times_test)
plt.title('number of vertices dependence of execution time')
plt.show()
