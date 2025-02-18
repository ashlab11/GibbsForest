from .Losses import *
from .LeafOrNode import LeafOrNode
import random
        
class Tree:
    def __init__(self, X, y, num_features_considering = 1, min_samples = 2, max_depth = 3,
                 loss_fn = LeastSquaresLoss(), initial_weight = 'parent', reg_lambda = 0):
        self.root = LeafOrNode(loss_fn.argmin(y), max_depth = max_depth, min_samples = min_samples,
                               initial_weight = initial_weight, loss_fn=loss_fn, reg_lambda = reg_lambda)
        self.loss_fn = loss_fn
        self.min_samples = min_samples
        self.max_depth = max_depth
        self.num_splits = 0
        self.features_considered = random.sample(range(X.shape[1]), num_features_considering)
        self.initial_weight = initial_weight
    def initial_splits(self, X, y, warmup_depth):
        """Function that does the initial splits, up to a warmup depth"""    
        num_splits = self.root.initial_split(X, y, warmup_depth, self.features_considered)
        self.num_splits += num_splits
        return num_splits
    def get_best_split(self, X, y, other_predictions, num_cols_other_prediction, eta):
        return self.root.get_best_split(X, y, other_predictions, num_cols_other_prediction, self.features_considered, eta) 
    def split(self, X, y):
        """Function that splits the tree"""
        self.root.split(X, y)
        self.num_splits += 1
    def predict(self, X_predict):
        return self.root.predict(X_predict)
    def revert_checks(self, X, y, other_predictions, alpha, ccp_alpha = 1e-6):
        """
        A pass to do reversion checks on all nodes in the tree.
        We'll do a DFS or BFS on the node structure.
        """
        reverts = self.root.revert(X, y, other_predictions=other_predictions, alpha=alpha, ccp_alpha = ccp_alpha)
        self.num_splits -= reverts
        return reverts
    
    def __repr__(self):
        return f"Tree with {self.num_splits} splits"