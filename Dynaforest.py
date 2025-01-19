import numpy as np
import random
from Tree import Tree
from line_profiler import profile
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import r2_score
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

class Dynatree(RegressorMixin, BaseEstimator):
    def __init__(self, n_trees = 10, window = 4, max_depth = 3, min_samples = 2, 
                 feature_subsampling_pct = 1, bootstrapping = True):
        self.n_trees = n_trees
        self.feature_subsampling_pct = feature_subsampling_pct
        self.bootstrapping = bootstrapping
        
        #Setting the window size
        if window == 'sqrt':
            self.window = int(np.sqrt(n_trees))
        elif int(window) == window:
            self.window = window
        else:
            raise ValueError("window must be an integer or sqrt")
        
        #Setting the max depth
        if max_depth is None:
            self.max_depth = np.inf
        elif isinstance(max_depth, int):
            self.max_depth = max_depth    
        else:
            raise ValueError('max_depth must be an integer or None')
        
        #Setting the min_samples
        if min_samples is None:
            self.min_samples = np.inf
        elif isinstance(min_samples, int):
            self.min_samples = min_samples
        else:
            raise ValueError('min samples must be an integer or none')
    
    def fit(self, X, y):
        X, y = check_X_y(X, y, accept_sparse=False)
        self.n_features_in_ = X.shape[1]  # for sklearn compliance
        self.num_features_considering = max(int(self.n_features_in_ * self.feature_subsampling_pct), 1)
        
        self._trees = []
        self._predictions = np.empty((self.n_trees, len(y)))
        
        X = np.array(X)
        y = np.array(y)
        bootstrapped_X = X
        bootstrapped_y = y
        
        for i in range(self.n_trees):
            if self.bootstrapping:
                bootstrapped_idx = np.random.choice(len(X), len(X), replace = True)
                bootstrapped_X = X[bootstrapped_idx]
                bootstrapped_y = y[bootstrapped_idx]
            
            
            features_to_consider = random.sample(range(self.n_features_in_), self.num_features_considering)
            
            """Creating all trees as stumps first"""
            #Note: must use bagging/subsampling to get better results here, should figure out how to do so?
            tree = Tree(bootstrapped_X, bootstrapped_y, max_depth=self.max_depth, min_samples = self.min_samples)
            tree.get_best_split(bootstrapped_X, bootstrapped_y, np.zeros(len(y)), 0, features_to_consider=features_to_consider)
            tree.split(bootstrapped_X, bootstrapped_y)
            self._trees.append(tree)
            self._predictions[i] = tree.get_training_predictions()
                
        while True:
            error_reductions = np.zeros(self.window)
            random_idx = random.sample(range(self.n_trees), self.window)
            predictions_to_consider = self._predictions[random_idx]
            
            if self.bootstrapping:
                bootstrapped_idx = np.random.choice(len(X), len(X), replace = True)
                bootstrapped_X = X[bootstrapped_idx]
                bootstrapped_y = y[bootstrapped_idx]
            
            features_to_consider = random.sample(range(self.n_features_in_), self.num_features_considering)
            
            
            for idx in range(self.window):
                tree = self._trees[random_idx[idx]]
                predictions_without_tree = np.delete(predictions_to_consider, idx, 0)
                if len(predictions_without_tree) == 0:
                    mean_predictions_without_tree = np.zeros(len(y))
                else:
                    mean_predictions_without_tree = np.mean(predictions_without_tree, axis = 0)
                error_reductions[idx] = tree.get_best_split(bootstrapped_X, bootstrapped_y, mean_predictions_without_tree, self.window - 1, 
                                                            features_to_consider)
            if max(error_reductions) <= 0:
                break
            best_error_idx = np.argmax(error_reductions)
            best_tree = self._trees[random_idx[best_error_idx]]
            best_tree.split(bootstrapped_X, bootstrapped_y)
            self._predictions[random_idx[best_error_idx]] = best_tree.get_training_predictions()
            
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
    
                
