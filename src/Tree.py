from .Losses import *
from .LeafOrNode import LeafOrNode
import random
        
class Tree:
    def __init__(self, X, y, num_features_considering = 1, min_samples = 2, max_depth = 3, eta = 0, 
                 loss_fn = LeastSquaresLoss()):
        self.root = LeafOrNode(loss_fn.argmin(y), max_depth = max_depth, min_samples = min_samples, eta = eta)
        self.min_samples = min_samples
        self.max_depth = max_depth
        self.num_splits = 0
        self.features_considered = random.sample(range(X.shape[1]), num_features_considering)
    def get_best_split(self, X, y, tree_predictions, other_predictions, num_cols_other_prediction):
        return self.root.get_best_split(X, y, tree_predictions, other_predictions, num_cols_other_prediction, self.features_considered) 
    def split(self, X, y):
        """Function that splits the tree"""
        self.root.split(X, y)
        self.num_splits += 1
    def predict(self, X_predict):
        return self.root.predict(X_predict)
    def __repr__(self):
        return f"Tree with {self.num_splits} splits"