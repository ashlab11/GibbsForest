from .Losses import *
from .LeafOrNode import LeafOrNode
import random
        
class Tree:
    def __init__(self, X, y, min_samples = 2, max_depth = 3,
                 loss_fn = LeastSquaresLoss(), initial_weight = 'parent'):
        self.root = LeafOrNode(loss_fn.argmin(y), max_depth = max_depth, min_samples = min_samples,
                               initial_weight = initial_weight, loss_fn=loss_fn)
        self.loss_fn = loss_fn
        self.min_samples = min_samples
        self.max_depth = max_depth
        self.num_splits = 0
        self.initial_weight = initial_weight
    def initial_splits(self, X, y, warmup_depth, num_features_considered, rng = np.random.default_rng()):
        """Function that does the initial splits, up to a warmup depth"""  
        features_considered = rng.choice(X.shape[1], num_features_considered, replace = False)
        num_splits = self.root.initial_split(X, y, warmup_depth, features_considered)
        self.num_splits += num_splits
    def get_best_split(self, X, y, other_predictions, features_considered, tree_weight, eta):
        return self.root.get_best_split(X, y, other_predictions, features_considered, tree_weight, eta) 
    def split(self, X, y):
        """Function that splits the tree"""
        self.root.split(X, y)
        self.num_splits += 1
    def predict(self, X_predict):
        return self.root.predict(X_predict)
    def __repr__(self):
        return f"Tree with {self.num_splits} splits"