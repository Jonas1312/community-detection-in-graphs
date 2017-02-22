#coding:utf-8

import numpy as np

def test_equality_communities(community1,community2,n_communities):
    nb_common_verticies = np.zeros(n_communities)
    for i in xrange(n_communities):
        community2 = (community2+1)%(n_communities)
        nb_common_verticies[i] = (community1==community2).sum()
    print(nb_common_verticies)
    print('{} well put verticies ({} vertices)'.format(nb_common_verticies.max(),len(community1)))
    return nb_common_verticies.max()
