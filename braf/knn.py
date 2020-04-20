import numpy as np 


def euclidean_distance_2(x0, x1):
    '''euclidean_distance_2 returns the *squared* euclidean distance between rows of x0 and x1'''
    # d[i,j] is x0[i] - x1[j]
    d = x0[:,None,:] - x1[None,:,:] 
    d2 = np.square(d)
    metric = np.sum(d2, axis=-1)
    return metric


class KNN:
    '''
    A K-NN implementatation that performs in O(n) time for each query

    Parameters:
    k (int): number of nearest neighbors 
    metric (str): the metric function. only Euclidean is supported at the moment
    '''
    def __init__(self, k, metric='euclidean'):
        self._k = k 
        if metric == 'euclidean':
            self._metric = euclidean_distance_2
    
    def get_neighbors(self, X, Z):
        '''
        compute the k nearest neighbors in Z of elements of X

        Parameters:
        X (ndarray)
        Z (ndarray)

        Returns:
        ndarray: an array of indices in Z, of shape (X.shape[0], k)
        '''

        # D[i,j] is the distance between X[i] and Z[j]
        D = self._metric(X, Z)        

        # k_closest[i] are the indices of the k closest elements of Z to X[i] 
        # note that they are *not* ordered
        k_closest = np.argpartition(D, self._k, axis=-1)[:, :self._k] 

        return k_closest 
    

if __name__ == '__main__':
    X = np.array([[0, 2], [2, 3]])
    Z = np.array([[0, 2], [1, 2], [3, 4], [-1, 0]])

    knn = KNN(3)
    print(Z[knn.get_neighbors(X, Z)])
