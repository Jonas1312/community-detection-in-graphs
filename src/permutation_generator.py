#!/usr/bin/env python
#coding:utf-8
"""
  Purpose:  Génération des permutations d'un ensemble fini
  Created:  22/02/2017
"""
# génération des permutations d'un ensemble fini
# par un procédé itératif utilisant les générateurs de python
# https://www.developpez.net/forums/d228599/general-developpement/algorithme-mathematiques/contribuez/faq-determination-combinaisons/#post1462943
# complexité en O(n_communities!) : convient pour un faible nombre de communautés

import numpy as np

def ins(X, i, L):
    """Fonction d'insertion"""
    R = L[:] # copie de L
    R.insert(i, X) # insertion de X à l'index i dans R
    return R

def turn(X, L):
    """Fait tourner X dans L à toutes les positions"""
    return [ins(X, i, L) for i in xrange(0, len(L) + 1)]

def permutations():
    """Générateur de permutations"""
    i, P = 0, [[]]
    yield P  # stoppe l'exécuton pour la reprendre à cet endroit précis
    while True:  # fait passer d'une génération à la suivante
        i = i + 1
        Q = [turn(i, X) for X in P]
        P = reduce(lambda x, y: x + y, Q)  # concaténation à répétition
        yield P  # stoppe l'exécution pour la reprendre à cet endroit précis

def permutation_calculator(communities_label_1, communities_label_2, n_communities):
    if len(communities_label_1) != len(communities_label_2):
        raise ValueError("communities_label_1 and communities_label_2 should be the same size")
    nb_common_vertices = np.array([])
    Perm = permutations()
    for i in range(0, n_communities+1):  # permutations d'un ensemble à n_communities éléments
        X = Perm.next()
    for permutation in X:
        communities_label_2_permuted = np.array([permutation[communities_label_2[i]]-1 for i in xrange(len(communities_label_1))])
        nb_common_vertices = np.append(nb_common_vertices, ((communities_label_1 == communities_label_2_permuted).sum()))
    return nb_common_vertices.max()
