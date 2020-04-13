import numpy as np


EPS = 1e-12

class BinaryDecisionNode:
    def __init__(self, X, y, node_depth=0, search_depth=7):
        '''On construction, the BinaryDecisionNode fits itself
        to the provided data. If node_depth > 0, it will create
        and fit two nodes on the results of this node'''

        # we assume data is in the form (samples, features)
        # and labels are a 1D binary vector
        assert len(X.shape) == 2
        assert len(y.shape) == 1

        gini0 = self.gini_index(y)
        best_delta = 0 
        for i_ft in range(X.shape[1]):
            x = X[:,i_ft]
            boundary, new_delta = self.max_delta_gini(
                    x=x, y=y, lo=np.min(x), hi=np.max(x),
                    depth=search_depth, gini0=gini0
                )
            print(i_ft, new_delta, best_delta)
            if new_delta > best_delta:
                best_delta = new_delta
                self.decision_feature = i_ft 
                self.decision_boundary = boundary 

        mask = self(X, descend=False) 
        self.right_val = y[mask].sum() / (y[mask].shape[0] + EPS)
        self.left_val = y[~mask].sum() / (y[~mask].shape[0] + EPS)

        if node_depth > 0:
            self.left = BinaryDecisionNode(
                    X[~mask], y[~mask], node_depth=node_depth-1, 
                    search_depth=search_depth
                )
            self.right = BinaryDecisionNode(
                    X[mask], y[mask], node_depth=node_depth-1, 
                    search_depth=search_depth
                )
        else:
            self.right = lambda X, y, descend: self.right_val
            self.left = lambda X, y, descend: self.left_val

    def __call__(self, X, y=None, descend=True):
        '''Infer the decision node on a dataset.
        If descend=True, we traverse the children
        until we hit leaves. Otherwise, just return a 
        boolean array of the decision at this node'''
        mask = X[:, self.decision_feature] > self.decision_boundary 

        if y is None:
            y = np.ones(X.shape[0]) # dummy
        yhat = np.ones_like(y)
        
        if descend:
            yhat[mask], _ = self.right(X[mask], y[mask], descend=True)
            yhat[~mask], _ = self.left(X[~mask], y[~mask], descend=True)
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
    def max_delta_gini(x, y, lo, hi, depth, gini0):
        '''recursive binary search to maximize
        the drop in gini index (with respect to
        gini0) in a specified range

        returns: decision boundary that maximizes delta gini'''
        print(lo, hi, depth, gini0)
        mid = (lo + hi) * 0.5
        if depth == 0:
            # we've reached the max recursion depth 
            # return the range midpoint 
            return mid, BinaryDecisionNode.delta_gini(x, y, mid, gini0) 
        mid_hi = (lo + hi) * 0.75 
        mid_lo = (lo + hi) * 0.25 
        delta_hi = BinaryDecisionNode.delta_gini(x, y, mid_hi, gini0)
        delta_lo = BinaryDecisionNode.delta_gini(x, y, mid_lo, gini0)
        # choose the next search range based on whether delta_hi or delta_lo is bigger 
        lo, hi = (lo, mid) if (delta_lo > delta_hi) else (mid, hi) 
        return BinaryDecisionNode.max_delta_gini(x, y, lo, hi, depth-1, gini0)

    @staticmethod 
    def gini_index(y):
        '''binary gini index'''
        p0 = (y == 0).sum() / (y.shape[0] + EPS) 
        return 1 - p0**2 - (1-p0)**2

    def __str__(self):
        s = f'Node(ft={self.decision_feature}, bndry={self.decision_boundary})'
        if isinstance(self.left, BinaryDecisionNode):
            s += f'\n\tleft: {str(self.left)}'
            s += f'\n\tright: {str(self.right)}'
        

if __name__ == '__main__':
    x = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]])
    y = np.array([0,       0,      0,       1,       0])
    node = BinaryDecisionNode(x, y, node_depth=1, search_depth=3) 
    print(str(node))
