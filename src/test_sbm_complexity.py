#!/usr/bin/env python
# coding:utf-8

from stochasticBlockModel import SBM
import matplotlib.pyplot as plt
from time import clock
import numpy as np
import csv

# ----------------------------------------------------------------------
# Measure execution time of sbm for several values of n_vertices
# Save results in csv file "..\execution_time.csv"
# This csv file can be imported in excel file "..\regression_temps_execution.xlsx" to compute complexity model
# ----------------------------------------------------------------------

cin = 15
cout = 5
n_communities = 2  # number of communities


n_vertices_test = range(100,2000,100)  # number of vertices for time measures
times_test = []

for n_vertices in n_vertices_test:
    probability_matrix = (1.0 / n_vertices) * (np.full((n_communities, n_communities), cout) + np.diag([cin - cout] * n_communities))  # matrix of edge probabilities
    time1 = clock()  # start time measure
    sbm = SBM(n_vertices, n_communities, probability_matrix)
    times_test.append((n_vertices, (clock() - time1)))

plt.plot(*zip(*times_test))
plt.xlabel("Nombre de noeuds")
plt.ylabel("Temps de generation")
plt.title('Generation time = f(Number of vertices)')
plt.show()

# Save execution time in csv file
csvfile = "..\execution_time.csv"

with open(csvfile, "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    for val in times_test:
        writer.writerow(val)
