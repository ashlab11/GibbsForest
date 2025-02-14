import numpy as np
import random
from .Tree import Tree
from .Losses import *
from .LeafOrNode import LeafOrNode
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

class GibbsForest(RegressorMixin, BaseEstimator):
    def __init__(self, loss_fn = LeastSquaresLoss(), n_trees = 10, max_depth = 3, min_samples = 2, 
                 feature_subsample = 1, row_subsample = 1, warmup_depth = 1, eta = 0.01, 
                 reg_lambda = 0, reg_gamma = 0, initial_weight = 'parent'):
        """
        Parameters:
        loss_fn: loss function to use for the trees (default is LeastSquaresLoss)
        n_trees: number of trees to use in the forest (default is 10)
        max_depth: maximum depth of the trees (default is 3)
        min_samples: minimum number of samples in a node to split (default is 2)
        feature_subsample: fraction of features to consider for each split (default is 1)
        row_subsample: fraction of rows to consider for each split (default is 1)
        warmup_depth: depth of each initial tree before using gibbs algorithm (default is 1)
        eta: learning rate (default is 0.01)
        reg_lambda: L2 regularization parameter (default is 0)
        TODO: reg_alpha: L1 regularization parameter (default is 0)
        reg_gamma: min loss reduction to make a split (default is 0)
        """
        
        self.loss_fn = loss_fn
        self.n_trees = n_trees
        self.eta = eta
        self.warmup_depth = warmup_depth
        self.feature_subsample = feature_subsample
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.row_subsample = row_subsample
        self.reg_lambda = reg_lambda
        self.reg_gamma = reg_gamma
        self.initial_weight = initial_weight
    
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
        self.num_features_considering = max(int(self.n_features_in_ * self.feature_subsample), 1)
        self.num_rows_considering = max(int(self.row_subsample * len(y)), 1)
        
        self._trees = []
        self._predictions = np.empty((self.n_trees, len(y)))
        
        X = np.array(X)
        y = np.array(y)
        bootstrapped_X = X.copy()
        bootstrapped_y = y.copy()

        #Initial tree creation and warmup levels        
        for i in range(self.n_trees):
            bootstrapped_idx = np.random.choice(len(X), self.num_rows_considering, replace = True)
            bootstrapped_X = X[bootstrapped_idx]
            bootstrapped_y = y[bootstrapped_idx]
                        
            """Creating all trees as stumps first"""
            #Note: must use bagging/subsample to get better results here, should figure out how to do so?
            tree = Tree(bootstrapped_X, bootstrapped_y, num_features_considering = self.num_features_considering, 
                        max_depth=self.max_depth, min_samples = self.min_samples, eta = self.eta)
            #Initial split -- no current tree-level predictions, and no predictions from any other splits
            #TODO: implement warmup depth here
            tree.get_best_split(bootstrapped_X, bootstrapped_y, np.zeros(self.num_rows_considering), 0)
            tree.split(bootstrapped_X, bootstrapped_y)            
            self._trees.append(tree)
            
            #TODO: implement feature importance here
            self._predictions[i] = tree.predict(X)
              
        #Round-robin -- with max_depth = N and initial depth D, we should have 2^N - 2^D splits
        for _ in range(2**self.max_depth - 2**self.warmup_depth):
            #Randomly selecting the permutation of trees to update
            tree_permutation = np.random.permutation(self.n_trees)
            
            #Selecting the batch of rows to consider for this round robin
            if self.row_subsample == 1:
                row_idx = slice(None) #Select all rows, quickly
            else:
                row_idx = np.random.choice(len(X), self.num_rows_considering, replace = False)
            X_batch = X[row_idx]
            y_batch = y[row_idx]

            #Getting predictions for the current batch of rows
            batch_predictions = [predictions[row_idx] for predictions in self._predictions]
            
            for tree_idx in tree_permutation:
                tree = self._trees[tree_idx]
                predictions_without_tree = np.delete(batch_predictions, tree_idx, 0)
                mean_predictions_without_tree = np.mean(predictions_without_tree, axis = 0)
                
                error_reduction, best_split = tree.get_best_split(X_batch, y_batch, mean_predictions_without_tree, self.n_trees - 1)
                if error_reduction > self.reg_gamma:
                    tree.split(X_batch, y_batch)
                    
                    #Predicts are based on the entire X, not X_batch, and new_predictions must be updated similarly
                    self._predictions[tree_idx] = tree.predict(X)
                    batch_predictions[tree_idx] = self._predictions[tree_idx][row_idx]
                    
                    #Updating feature importances and splits
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
