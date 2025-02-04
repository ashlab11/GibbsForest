import numpy as np
import random
from Tree import Tree
from line_profiler import profile
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import r2_score
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

class Dynatree(RegressorMixin, BaseEstimator):
    def __init__(self, n_trees = 10, window = 4, max_depth = 3, min_samples = 2, 
                 feature_subsampling_pct = 1, bootstrapping = True, delta = 0):
        self.n_trees = n_trees
        self.delta = delta
        self.feature_subsampling_pct = feature_subsampling_pct
        self.bootstrapping = bootstrapping
        self.window = window
        self.max_depth = max_depth
        self.min_samples = min_samples
    
    def fit(self, X, y):
        X, y = check_X_y(X, y, accept_sparse=False)
        self.n_features_in_ = X.shape[1]  # for sklearn compliance
        
        #Setting the window size
        if self.window == 'sqrt':
            self.window = int(np.sqrt(self.n_trees))
        elif self.window == 'log2':
            self.window = int(np.log2(self.n_trees))
        elif int(self.window) == self.window:
            self.window = self.window
        else:
            raise ValueError("window must be an integer or sqrt")
        
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
        times_without_improvement = 0
        
        for i in range(self.n_trees):
            if self.bootstrapping:
                bootstrapped_idx = np.random.choice(len(X), len(X), replace = True)
                bootstrapped_X = X[bootstrapped_idx]
                bootstrapped_y = y[bootstrapped_idx]
            
                        
            """Creating all trees as stumps first"""
            #Note: must use bagging/subsampling to get better results here, should figure out how to do so?
            tree = Tree(bootstrapped_X, bootstrapped_y, num_features_considering = self.num_features_considering, 
                        max_depth=self.max_depth, min_samples = self.min_samples, delta = self.delta)
            #Initial split -- no current tree-level predictions, and no predictions from any other splits
            tree.get_best_split(bootstrapped_X, bootstrapped_y, np.zeros(len(y)), np.zeros(len(y)), 0)
            tree.split(bootstrapped_X, bootstrapped_y)
            self._trees.append(tree)
            
            #Predictions should be based on X, not bootstrapped_X
            #TODO: implement feature importance here
            self._predictions[i] = tree.predict(X)
                
        while True:
            error_reductions = []
            idxs_to_consider = random.sample(range(self.n_trees), self.window)
            predictions_to_consider = self._predictions[idxs_to_consider]
            splitting_cols = []
                        
            for considering_idx, overall_idx in enumerate(idxs_to_consider):
                tree = self._trees[overall_idx]
                
                #Get the predictions of everything else as well as current preds from the tree
                predictions_without_tree = np.delete(predictions_to_consider, considering_idx, 0)
                tree_level_predictions = predictions_to_consider[considering_idx]
                
                if len(predictions_without_tree) == 0:
                    mean_predictions_without_tree = np.zeros(len(y))
                else:
                    mean_predictions_without_tree = np.mean(predictions_without_tree, axis = 0)
                        
                error_reduction, best_split = tree.get_best_split(X, y, tree_level_predictions, mean_predictions_without_tree, self.window - 1)
                error_reductions.append(error_reduction)
                splitting_cols.append(best_split)
                
            if max(error_reductions) <= 0:
                times_without_improvement += 1
                if times_without_improvement >= 10:
                    break
                continue
            best_error_idx = np.argmax(error_reductions)
            best_col = splitting_cols[best_error_idx]
            
            self.feature_importances_[best_col] += max(error_reductions)
            self.feature_splits[best_col] += 1
            
            best_tree = self._trees[idxs_to_consider[best_error_idx]]
            best_tree.split(X, y)
            self._predictions[idxs_to_consider[best_error_idx]] = best_tree.predict(X)
            
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
