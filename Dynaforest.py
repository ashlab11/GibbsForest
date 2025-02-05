import numpy as np
import random
from Tree import Tree
from line_profiler import profile
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import r2_score
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

class Dynatree(RegressorMixin, BaseEstimator):
    def __init__(self, n_trees = 10, max_depth = 3, min_samples = 2, 
                 feature_subsampling_pct = 1, warmup_depth = 1, delta = 0):
        self.n_trees = n_trees
        self.delta = delta
        self.warmup_depth = warmup_depth
        self.feature_subsampling_pct = feature_subsampling_pct
        self.max_depth = max_depth
        self.min_samples = min_samples
    
    def fit(self, X, y):
        X, y = check_X_y(X, y, accept_sparse=False)
        self.n_features_in_ = X.shape[1]  # for sklearn compliance
        
        if self.n_trees is None or self.n_trees < 2:
            raise ValueError('n_trees must be an integer greater than 1')
        
        #Setting the max depth
        if self.max_depth is None:
            self.max_depth = np.inf
        elif isinstance(self.max_depth, int):
            self.max_depth = self.max_depth    
        else:
            raise ValueError('max_depth must be an integer or None')
        
        #Setting the min_samples
        if self.min_samples is None:
            self.min_samples = np.inf
        elif isinstance(self.min_samples, int):
            self.min_samples = self.min_samples
        else:
            raise ValueError('min samples must be an integer or none')
        
        self.feature_importances_ = np.zeros(self.n_features_in_)
        self.feature_splits = np.zeros(self.n_features_in_)
        self.num_features_considering = max(int(self.n_features_in_ * self.feature_subsampling_pct), 1)
        
        self._trees = []
        self._predictions = np.empty((self.n_trees, len(y)))
        
        X = np.array(X)
        y = np.array(y)
        bootstrapped_X = X.copy()
        bootstrapped_y = y.copy()

        #Initial tree creation and warmup levels        
        for i in range(self.n_trees):
            bootstrapped_idx = np.random.choice(len(X), len(X), replace = True)
            bootstrapped_X = X[bootstrapped_idx]
            bootstrapped_y = y[bootstrapped_idx]
            
                        
            """Creating all trees as stumps first"""
            #Note: must use bagging/subsampling to get better results here, should figure out how to do so?
            tree = Tree(bootstrapped_X, bootstrapped_y, num_features_considering = self.num_features_considering, 
                        max_depth=self.max_depth, min_samples = self.min_samples, delta = self.delta)
            #Initial split -- no current tree-level predictions, and no predictions from any other splits
            #TODO: implement warmup depth here
            tree.get_best_split(bootstrapped_X, bootstrapped_y, np.zeros(len(y)), np.zeros(len(y)), 0)
            tree.split(bootstrapped_X, bootstrapped_y)
            self._trees.append(tree)
            
            #Predictions should be based on X, not bootstrapped_X
            #TODO: implement feature importance here
            self._predictions[i] = tree.predict(X)
              
        #Round-robin -- with max_depth = N and initial depth D, we should have 2^N - 2^D splits
        for _ in range(2**self.max_depth - 2**self.warmup_depth):
            for tree_idx, tree in enumerate(self._trees):
                predictions_without_tree = np.delete(self._predictions, tree_idx, 0)
                tree_level_predictions = self._predictions[tree_idx]
                mean_predictions_without_tree = np.mean(predictions_without_tree, axis = 0)
                
                error_reduction, best_split = tree.get_best_split(X, y, tree_level_predictions, mean_predictions_without_tree, self.n_trees - 1)
                if error_reduction > 0:
                    tree.split(X, y)
                    self._predictions[tree_idx] = tree.predict(X)
                    self.feature_importances_[best_split] += error_reduction
                    self.feature_splits[best_split] += 1
        return self 
        

    def predict(self, X_predict):
        X_predict = check_array(X_predict, accept_sparse=False)
        check_is_fitted(self, 'n_features_in_')  # raises NotFittedError if missing
        print(f"This forest has a total of {sum([tree.num_splits for tree in self._trees])} splits")
        print(f"This forest has a total of {len(self._trees)} trees")
        
        """We want to go through each tree and combine predictions"""
        predictions = []
        for tree in self._trees:
            predictions.append(tree.predict(X_predict))
        return np.mean(predictions, axis = 0)
