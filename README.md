# CommunityDetection

Community detection in graphs/networks project
Stochastic block model implementation, spectral clustering with the bethe hessian and comparison with other methods (adjacency matrix, Laplacian matrix, modularity matrix)

## Stochastic block model
From [Wikipedia](https://en.wikipedia.org/wiki/Stochastic_block_model):

The stochastic block model takes the following parameters:

* The number *n* of vertices
* a partition of the vertex set {1, ..., n} into disjoint subsets {C_1, ..., C_r} called communities
* a symmetric r x r matrix P of edge probabilities.
The edge set is then sampled at random as follows: any two vertices u in C_i and v in C_j are connected by an edge with probability P_ij.

## Requirements
  - Python 2.7
  - Networkx
  - Numpy
  
## Installation (Windows)
Open command prompt in admin and type the commands:
```cmd
pip install numpy
pip install networkx
```