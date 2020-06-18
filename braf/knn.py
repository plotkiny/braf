#!/usr/bin/env python3

import numpy as np


def cosine_similarity(x0, x1):
    '''For 2 vectors calculate cosine similarity'''
    dot_product = np.dot(x0, np.transpose(x1))
    norm_x0 = np.linalg.norm(x0)
    norm_x1 = np.linalg.norm(x1)
    return dot_product / (norm_x0 * norm_x1)


def euclidean_distance_2(x0, x1):
    '''euclidean_distance_2 returns the *squared* euclidean distance between rows of x0 and x1'''
    # d[i,j] is x0[i] - x1[j]
    d = x0[:, None, :] - x1[None, :, :]
    d2 = np.square(d)
    metric = np.sum(d2, axis=-1)
    return metric


class KNN:
    '''
    A K-NN implementatation that performs in O(n) time for each query
    '''

    @staticmethod
    def get_neighbors(X, Z, k=20, metric='euclidean'):
        '''
        compute the k nearest neighbors in Z of elements of X

        Parameters:

        - X (ndarray)
        - Z (ndarray)
        - k (int): number of nearest neighbors
        - metric (str): the metric function. only Euclidean is supported at the moment

        Returns:

        - ndarray: an array of indices in Z, of shape (X.shape[0], k)
        '''

        # allow metric to be changed
        if metric == 'euclidean':
            _metric = euclidean_distance_2
        elif metric == "cosine":
            _metric = cosine_similarity

        # D[i,j] is the distance between X[i] and Z[j]
        D = _metric(X, Z)

        # k_closest[i] are the indices of the k closest elements of Z to X[i]
        # note that they are *not* ordered
        k_closest = np.argpartition(D, k, axis=-1)[:, :k]

        return k_closest


if __name__ == '__main__':
    X = np.array([[0, 2], [2, 3]])
    Z = np.array([[0, 2], [1, 2], [3, 4], [-1, 0]])
    print(Z[KNN.get_neighbors(X, Z, k=3)])
