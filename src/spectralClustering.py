#!/usr/bin/env python
#coding:utf-8
"""
  Purpose:  Spectral clustering algorithm
  Created:  12/02/2017
"""

# ----------------------------------------------------------------------
# Spectral clustering
# ----------------------------------------------------------------------
from matrices import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import networkx as nx


def Kmeans_methode(eigvals,eigvects,G,n_vertices,labels,color_map,usedMatrix):
    '''
    Plot the histogram of eig values, the eigen vectors corresponding to the two smallest eigenvalues
    and networkx graph of communities computed with Kmeans
    :param eigvals: numpy vector
    :param eigvects: numpy matrix
    :param G: networx graph
    :param n_vertices: int
    :param labels: labels of vertices
    :param color_map: colors for communities
    :param usedMatrix: string (name of the matrix)
    :return:
    '''
    plt.title("Histogram of {} eigenvalues".format(usedMatrix))
    n_clusters = 2
    plt.hist(eigvals, bins=100)  # plot histogram

    indices = eigvals.argsort()[:n_clusters]  # find the two smallest eigenvalues indices
    W = np.column_stack((eigvects[:, indices[0]], eigvects[:, indices[1]]))
    plt.figure()
    plt.title("Eigenvectors corresponding to the two smallest eigenvalues")
    plt.plot(W[:, 0], W[:, 1], 'o', markersize=5)

    kmeans = KMeans(n_clusters=n_clusters).fit(W)  # kmeans
    plt.figure()
    plt.title("K-means")
    for i in xrange(n_clusters):
        ds = W[np.where(kmeans.labels_ == i)]
        plt.plot(ds[:, 0], ds[:, 1], color=color_map[i], marker='o', markersize=5, ls='')

    if n_vertices > 300:
        print("Can't draw if number of vertices is too big")
    else:
        plt.figure()
        plt.title("Detected communities with the {}".format(usedMatrix))
        nx.draw(G, labels=labels, node_color=color_map[~kmeans.labels_], font_size=10)
    plt.show()


def spectralClustering_ModularityMatrix(adjacency_matrix,G,n_vertices,labels,color_map):
    """
    Compute the eig values and eig vectors of the modularity matrix and execute Kmeans_methode
    :param adjacency_matrix: numpy matrix
    :param G: networkx graph
    :param n_vertices: int
    :param labels: labels of vertices
    :param color_map: colors of communities
    :return:
    """
    eigvals, eigvects = np.linalg.eig(ModularityMatrix(
        adjacency_matrix))  # eigvects[:,i] is the eigenvector corresponding to the eigenvalue eigvals[i]
    Kmeans_methode(eigvals,eigvects,G,n_vertices,labels,color_map,'Modularity Matrix')

def spectralClustering_LaplacianMatrix(adjacency_matrix,G,n_vertices,labels,color_map):
    """
    Compute the eig values and eig vectors of the Laplacian matrix and execute Kmeans_methode
    :param adjacency_matrix: numpy matrix
    :param G: networkx graph
    :param n_vertices: int
    :param labels: labels of vertices
    :param color_map: colors of communities
    :return:
    """
    eigvals, eigvects = np.linalg.eig(LaplacianMatrix(
        adjacency_matrix))  # eigvects[:,i] is the eigenvector corresponding to the eigenvalue eigvals[i]
    Kmeans_methode(eigvals,eigvects,G,n_vertices,labels,color_map,'Laplacian Matrix')

def spectralClustering_BetheHessianMatrix(adjacency_matrix,G,n_vertices,labels,color_map):
    """
    Compute the eig values and eig vectors of the Bethe-Hessian matrix and execute Kmeans_methode
    :param adjacency_matrix: numpy matrix
    :param G: networkx graph
    :param n_vertices: int
    :param labels: labels of vertices
    :param color_map: colors of communities
    :return:
    """
    eigvals, eigvects = np.linalg.eig(BetheHessian(
        adjacency_matrix))  # eigvects[:,i] is the eigenvector corresponding to the eigenvalue eigvals[i]
    Kmeans_methode(eigvals,eigvects,G,n_vertices,labels,color_map,'Bethe-Hessian Matrix')