from .Losses import *
from .LeafOrNode import LeafOrNode
import random
        
class Tree:
    def __init__(self, init, min_samples = 2, max_depth = 3,
                 loss_fn = LeastSquaresLoss(), initial_weight = 'parent', hist_splitter = None):
        self.root = LeafOrNode(init, hist_splitter = hist_splitter, max_depth = max_depth, min_samples = min_samples,
                               initial_weight = initial_weight, loss_fn=loss_fn)
        self.loss_fn = loss_fn
        self.min_samples = min_samples
        self.max_depth = max_depth
        self.num_leaves = 1 #Starts at 1 and increases
        self.initial_weight = initial_weight
        self.hist_splitter = hist_splitter
    def initial_splits(self, row_idxs, warmup_depth, num_features_considered, rng = np.random.default_rng()):
        """Function that does the initial splits, up to a warmup depth"""  
        features_considered = rng.choice(self.hist_splitter.n_features, num_features_considered, replace = False)
        num_leaves = self.root.initial_split(row_idxs, warmup_depth, features_considered)
        self.num_leaves += num_leaves
    def get_best_split(self, row_idxs, other_predictions, features_considered, tree_weight, eta):
        return self.root.get_best_split(row_idxs, other_predictions, features_considered, tree_weight, eta) 
    def split(self):
        """Function that splits the tree"""
        self.root.split()
        self.num_leaves += 1
    def predict(self, X_predict):
        return self.root.predict(X_predict)
    def __repr__(self):
        return f"Tree with {self.num_leaves} leaves"