# CommunityDetection

Community detection in graphs/networks project

Stochastic block model implementation, spectral clustering with the bethe hessian and comparison with other methods (adjacency matrix, Laplacian matrix, modularity matrix)

## Stochastic block model
From [Wikipedia](https://en.wikipedia.org/wiki/Stochastic_block_model):

The stochastic block model takes the following parameters:

* The number *n* of vertices
* a partition of the vertex set {1, ..., n} into disjoint subsets {C1, ..., Cr} called communities
* a symmetric r x r matrix P of edge probabilities.
The edge set is then sampled at random as follows: any two vertices u in Ci and v in Cj are connected by an edge with probability Pij.

## Bethe Hessian matrix
[arXiv.org](https://arxiv.org/abs/1406.1880)

## Spectral clustering
[Wikipedia](https://en.wikipedia.org/wiki/Spectral_clustering)


## Requirements
  - Python 2.7
  - Networkx
  - Numpy
  - Sklearn
  - Scipy