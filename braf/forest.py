

import numpy as np
import tqdm
from functools import partial 

EPS = 1e-12


# put this in top-level scope so it can be pickled 
def _return_const(*args, **kwargs):
    return kwargs['val'] 


class BinaryDecisionNode:
    def __init__(self, X, y, node_depth=0, n_search_pts=100, max_features=None):
        '''
        Binary decision node.

        On construction, the BinaryDecisionNode fits itself
        to the provided data. If node_depth > 0, it will create
        and fit two nodes on the results of this node

        Parameters:

        - X (ndarray): features
        - y (ndarray): binary labels
        - node_depth (int): number of nodes to descend 
        - n_search_pts (int): number of points in the grid used to search for an
                              optimal decision threshold 
        - max_features (int): maximum number of features to optimize over at each node
        '''

        # we assume data is in the form (samples, features)
        # and labels are a 1D binary vector
        assert len(X.shape) == 2
        assert len(y.shape) == 1

        self.decision_boundary = None
        self.decision_feature = None

        # randomly sample at most max_features for fitting
        if max_features is None:
            features = np.arange(X.shape[1])
        else:
            features = np.random.permutation(X.shape[1])[:max_features]

        # gini index of the data fed into this node
        gini0 = self.gini_index(y)
        best_delta = 0 
        for i_ft in features:
            # iterate over the possible features, find the best 
            # boundary and the corresponding gini index. pick the 
            # choice of i_ft and boundary that maximizes the drop
            # from gini0 (new_delta)
            x = X[:,i_ft]
            boundary, new_delta = self.max_delta_gini(
                    x=x, y=y, lo=np.min(x), hi=np.max(x),
                    n_search_pts=n_search_pts, gini0=gini0
                )
            if new_delta > best_delta:
                best_delta = new_delta
                self.decision_feature = i_ft 
                self.decision_boundary = boundary 

        leaf = True # whether this node is a leaf in the tree 
        if self.decision_boundary is not None:
            # we found a valid decision boundary

            # run inference on this node to split the dataset. 
            mask = self(X, descend=False) 

            # save the mean values of the left and right splits.
            # not strictly needed if this is not a leaf, but it's
            # useful for debugging. could also be useful if we ever
            # prune the tree.
            self.right_val = y[mask].sum() / (y[mask].shape[0] + EPS)
            self.left_val = y[~mask].sum() / (y[~mask].shape[0] + EPS)

            if node_depth > 0:
                # if we haven't reached the max node depth, fit left
                # and right nodes:
                self.left = BinaryDecisionNode(
                        X[~mask], y[~mask], node_depth=node_depth-1, 
                        n_search_pts=n_search_pts
                    )
                self.right = BinaryDecisionNode(
                        X[mask], y[mask], node_depth=node_depth-1, 
                        n_search_pts=n_search_pts
                    )
                leaf = False # has children -> not a leaf 
        else:
            # no valid decision boundary, so left = right. use the mean
            # of the current dataset as the prediction for this node
            self.left_val = self.right_val = np.mean(y) 

        if leaf:
            # if no children, make the 'children' the mean value in each category
            self.right = partial(_return_const, val=self.right_val)
            self.left = partial(_return_const, val=self.left_val)

    def __call__(self, X, y=None, descend=True):
        '''
        infer the decision node on a dataset.

        if descend=true, we traverse the children
        until we hit leaves. otherwise, just return a 
        boolean array of the decision at this node

        Parameters: 

        - X (ndarray): features
        - y (ndarray): labels, optional 
        - descend (boolean): should we descend down into children nodes or not

        Returns:

        - ndarray: if descend, we guarantee to end at a leaf, so return the prediction
                 of leaf nodes. if not descend, just return a 0-1 array containing the
                 the decision of this node
        '''

        if y is None:
            y = np.ones(X.shape[0]) # dummy
        yhat = np.ones_like(y)

        if self.decision_boundary is None:
            # this node is a dummy
            return self.left() * yhat

        mask = X[:, self.decision_feature] > self.decision_boundary 
        
        if descend:
            yhat[mask] = self.right(X[mask], y[mask], descend=True)
            yhat[~mask] = self.left(X[~mask], y[~mask], descend=True)
            return yhat 
        else:
            return mask 

    @staticmethod
    def delta_gini(x, y, boundary, gini0):
        '''compute the drop in gini index (with
        respect to gini0) if we place a decision
        boundary at boundary'''
        mask = x > boundary 
        gini_hi = BinaryDecisionNode.gini_index(y[mask])
        gini_lo = BinaryDecisionNode.gini_index(y[~mask])
        frac = mask.astype(int).sum() / mask.shape[0] 
        delta = gini0 - (frac * gini_hi) - ((1 - frac) * gini_lo)
        return delta

    @staticmethod 
    def max_delta_gini(x, y, lo, hi, n_search_pts, gini0):
        '''grid search to maximize
        the drop in gini index (with respect to
        gini0) in a specified range

        Returns: 
        
        - decision boundary that maximizes delta gini, delta gini'''
        search_pts = np.linspace(lo, hi, n_search_pts)
        best_delta = 0 
        best_threshold = lo 
        for bndry in search_pts:
            new_delta = BinaryDecisionNode.delta_gini(x, y, bndry, gini0)
            if new_delta > best_delta:
                best_delta = new_delta 
                best_threshold = bndry

        return best_threshold, best_delta

    @staticmethod 
    def gini_index(y):
        '''binary gini index'''
        p0 = (y == 0).sum() / (y.shape[0] + EPS) 
        return 1 - p0**2 - (1-p0)**2

    def __str__(self):
        '''
        recursively write a descriptive string for this node
        and its children. useful for debugging 
        '''
        def parse_child(s):
            s = s.split('\n')
            s = s[:1] + ['        ' + ss for ss in s[1:]]
            return '\n'.join(s)
        s = ''
        if self.decision_feature is None:
            s += f'Node(pred={self.left()})'
        else:
            s += f'Node(ft={self.decision_feature}, bndry={self.decision_boundary}'
            if isinstance(self.left, BinaryDecisionNode):
                s += f',\n     L: {parse_child(str(self.left))}'
                s += f',\n     R: {parse_child(str(self.right))}'
            else:
                s += f', L={str(self.left())}'
                s += f', R={str(self.right())}'
            s += ')'
        return s


class BinaryDecisionTree:
    '''
    Binary decision tree. 

    Parameters:

    - node_depth (int): maximum depth of the tree 
    - n_search_pts (int): number of points in the search grid 
    - max_features_per_node (int): maximum number of features each node is allowed to use 
    '''
    def __init__(self, node_depth, n_search_pts=100, max_features_per_node=None):
        self.node_depth = node_depth 
        self.max_features = max_features_per_node
        self.n_search_pts = n_search_pts
        self.root = None 

    def fit(self, X, y):
        '''
        Fit the tree

        Parameters:

        - X (ndarray): features 
        - y (ndaray): labels 
        '''

        self.root = BinaryDecisionNode(X, y, self.node_depth, 
                                       n_search_pts=self.n_search_pts,
                                       max_features=self.max_features)

    def predict(self, X):
        '''
        Run inference 

        Parameters 

        - X (ndarray): features 
        '''

        assert (self.root is not None), ('Trying to run predict on a tree that has not been fit!')
        return self.root(X)

    def __str__(self):
        return str(self.root)


class RandomForest:
    '''
    Random forest

    Parameters:

    - n_trees (int): number of trees 
    - bagging_frac (int): fraction of training set used for training each tree  
    - node_depth (int): maximum tree depth 
    - n_search_pts (int): number of points used to search for optimal decision threshold 
    - max_features_per_node (int): maximum features each node is allowed to use 
    '''
    def __init__(self, n_trees, bagging_frac, node_depth, n_search_pts, max_features_per_node):
        self.trees = [BinaryDecisionTree(node_depth, n_search_pts, max_features_per_node) 
                      for _ in range(n_trees)]
        self.bagging_frac = bagging_frac 

    def fit(self, X, y):
        '''
        Fit the forest 

        Parameters:

        - X (ndarray): features 
        - y (ndaray): labels 
        '''
        for tree in self.trees: 
            N = 0
            while N == 0:
                mask = np.random.binomial(1, size=X.shape[0], p=self.bagging_frac).astype(bool)
                N = mask.shape[0] # make sure we never sample an empty set
            tree.fit(X[mask, :], y[mask])

    def predict(self, X):
        '''
        Run inference 

        Parameters 
        
        - X (ndarray): features 
        '''

        return np.mean([tree.predict(X) for tree in self.trees], axis=0)

    def __str__(self):
        return '\n'.join([str(t) for t in self.trees])
        

if __name__ == '__main__':
    x = np.array([[0, 1], [1, 2], [2, 3], [2,3], [3, 4], [4, 5]])
    y = np.array([0,       0,      0,       1,     1,       0])
    tree = BinaryDecisionTree(node_depth=5)
    tree.fit(x, y)
    print(str(tree))
    yhat = tree.predict(x)
    print(np.stack([y, yhat], axis=-1))
    print()

    forest = RandomForest(6, 0.5, 1, 1)
    forest.fit(x, y)
    print(str(forest))
    yhat = forest.predict(x)
    print(np.stack([y, yhat], axis=-1))
