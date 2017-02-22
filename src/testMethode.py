# coding:utf-8

import numpy as np


def test_equality_communities(community1, community2, n_communities):
    """
    compute the consistency between two communities labels
    :param community1:np.array (label of communities)
    :param community2:np.array (label of communities)
    :param n_communities:int
    :return: number of vertices put in well community
    """
    nb_common_vertices = np.zeros(n_communities)
    for i in xrange(n_communities):
        community2 = (community2 + 1) % n_communities
        nb_common_vertices[i] = (community1 == community2).sum()
    print('{} well put vertices ({} vertices)'.format(nb_common_vertices.max(), len(community1)))
    return nb_common_vertices.max()
