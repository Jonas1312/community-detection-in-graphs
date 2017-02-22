# coding=utf-8
# génération des permutations d'un ensemble fini
# par un procédé itératif utilisant les générateurs de python
# inspiré par code trouvé sur https://www.developpez.net/forums/d228599/general-developpement/algorithme-mathematiques/contribuez/faq-determination-combinaisons/#post1462943
# complexité en O(n_communities!) : convient pour un faible nombre de communautés

import numpy as np

# fonction d'insertion
def ins(X, i, L):
    R = L[:]  # copie de L
    R.insert(i, X)  # insertion de X à l'index i dans R
    return R


# fait tourner X dans L à toutes les positions
def turn(X, L):
    return [ins(X, i, L) for i in range(0, len(L) + 1)]


# un générateur
def permutations():
    i, P = 0, [[]]
    yield P  # stoppe l'exécuton pour la reprendre à cet endroit précis
    while True:  # fait passer d'une génération à la suivante
        i = i + 1
        Q = [turn(i, X) for X in P]
        P = reduce(lambda x, y: x + y, Q)  # concaténation à répétition
        yield P  # stoppe l'exécution pour la reprendre à cet endroit précis


def test_methode(communities_label_1, communities_label_2, n_communities, n_vertices):
    nb_common_vertices = np.array([])
    Perm = permutations()
    for i in range(0, n_communities + 1):  # permutations d'un ensemble à n_communities éléments
        X = Perm.next()
    for permutation in X:
        communities_label_2_permuted = np.array([permutation[communities_label_2[i]]-1 for i in xrange(n_vertices)])
        nb_common_vertices=np.append(nb_common_vertices,((communities_label_1 == communities_label_2_permuted).sum()))
    print('{} well put vertices ({} vertices)'.format(nb_common_vertices.max(), n_vertices))


