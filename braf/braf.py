from .knn import KNN
from .forest import RandomForest 
import numpy as np


class BRAF:
    def __init__(self, K, s, p, minority_label, *args, **kwargs): 
        self.knn = KNN(K)
        self.rf1 = RandomForest(int(s * (1-p)), *args, **kwargs)
        self.rf2 = RandomForest(int(s * p), *args, **kwargs)
        self.p = p 
        self.min_label = minority_label 

    def fit(self, X, y):
        critical_idcs = self.knn.get_neighbors(X[y==self.min_label], X[y!=self.min_label]) 
        # critical_idcs contains duplicates, so flatten and remove
        critical_idcs = np.array(list(set(critical_idcs.reshape(-1))))

        # fit on the whole dataset 
        self.rf1.fit(X, y)
        # fit on the critical area 
        self.rf2.fit(X[critical_idcs, :], y[critical_idcs])

    def predict(self, X):
        # return a weighted sum of the predictions 
        yhat1 = self.rf1.predict(X)
        yhat2 = self.rf2.predict(X)
        return ((1-self.p) * yhat1) + (self.p * yhat2)

    def __str__(self):
        s = '=============== RF1 ================='
        s += str(self.rf1)
        s += '\n\n=============== RF2 ================='
        s += str(self.rf2)
