import numpy as np
import random
from .Tree import Tree
from .Losses import *
from .LeafOrNode import LeafOrNode
from .ParamErrors import check_params
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

class GibbsForest(RegressorMixin, BaseEstimator):
    def __init__(self, loss_fn = LeastSquaresLoss(), n_trees = 10, max_depth = 3, min_samples = 2, 
                 feature_subsample = 1, row_subsample = 1, warmup_depth = 1, eta = 0.01, 
                 reg_lambda = 0, reg_gamma = 0, initial_weight = 'parent', 
                 eta_decay = 1, dropout = 0, ccp_alpha = None):
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
        initial_weight: initial weight of the tree (default is 'parent')
        eta_decay: learning rate decay (default is 1)
        dropout: probability of skipping a tree update (default is 0)
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
        self.eta_decay = eta_decay
        self.dropout = dropout
        self.ccp_alpha = ccp_alpha
    
    def fit(self, X, y):
        X, y = check_X_y(X, y, accept_sparse=False)
        self.n_features_in_ = X.shape[1]  # for sklearn compliance
        
        #Checking parameters
        self.loss_fn, self.n_trees, self.max_depth, self.min_samples, self.feature_subsample, self.row_subsample, self.warmup_depth, self.eta, self.reg_lambda, self.reg_gamma, self.initial_weight, self.eta_decay, self.dropout, self.ccp_alpha = check_params(
            self.loss_fn, self.n_trees, self.max_depth, self.min_samples, self.feature_subsample, self.row_subsample, self.warmup_depth, self.eta, self.reg_lambda, self.reg_gamma, self.initial_weight, self.eta_decay, self.dropout, self.ccp_alpha)
        
        self.feature_importances_ = np.zeros(self.n_features_in_)
        self.feature_splits = np.zeros(self.n_features_in_)
        self.num_features_considering = max(int(self.n_features_in_ * self.feature_subsample), 1)
        self.num_rows_considering = max(int(self.row_subsample * len(y)), 1)
        self.reversions = 0
        
        self._trees = []
        self._predictions = np.empty((self.n_trees, len(y)))
        
        X = np.array(X)
        y = np.array(y)
        bootstrapped_X = X.copy()
        bootstrapped_y = y.copy()

        #---- INITIAL TREE CREATION, UP TO WARMUP DEPTH ----#     
        for i in range(self.n_trees):
            bootstrapped_idx = np.random.choice(len(X), self.num_rows_considering, replace = True)
            bootstrapped_X = X[bootstrapped_idx]
            bootstrapped_y = y[bootstrapped_idx]
            
            tree = Tree(bootstrapped_X, bootstrapped_y, num_features_considering = self.num_features_considering, 
                        max_depth=self.max_depth, min_samples = self.min_samples, initial_weight = self.initial_weight, 
                        loss_fn=self.loss_fn, reg_lambda = self.reg_lambda)
            #Initial split -- no current tree-level predictions, and no predictions from any other splits
            tree.initial_splits(bootstrapped_X, bootstrapped_y, self.warmup_depth)
            self._trees.append(tree)
            
            #TODO: implement feature importance here
            self._predictions[i] = tree.predict(X)
            
        #---- GIBBS TREE CREATION ----#
        #Round-robin -- with max_depth = N and initial depth D, we should have 2^N - 2^D splits
        while sum([tree.num_splits for tree in self._trees]) < len(self._trees) * 2**(self.max_depth - 0.5):
        #for _ in range(2**self.max_depth - 2**self.warmup_depth):
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
                # ---- Dropout condition: skip updating this tree with probability self.dropout ----
                if random.random() < self.dropout:
                    continue  # skip update, move to the next tree

                tree = self._trees[tree_idx]
                predictions_without_tree = np.delete(batch_predictions, tree_idx, 0)
                mean_predictions_without_tree = np.mean(predictions_without_tree, axis = 0)
                
                error_reduction, best_split = tree.get_best_split(X_batch, y_batch, mean_predictions_without_tree, self.n_trees - 1, 
                                                                  self.eta)
                if error_reduction > self.reg_gamma: #Only split if gain is above gamma
                    tree.split(X_batch, y_batch)
                    
                    #Predictions are based on the entire X, not X_batch, and new_predictions must be updated similarly
                    self._predictions[tree_idx] = tree.predict(X)
                    batch_predictions[tree_idx] = self._predictions[tree_idx][row_idx]
                    
                    #Updating feature importances and splits
                    self.feature_importances_[best_split] += error_reduction
                    self.feature_splits[best_split] += 1
            
            self.eta *= self.eta_decay
            
             # On-the-fly reversion pass (optional)
            if self.ccp_alpha is not None:
                # We'll do a pass over all trees, re-checking splits
                for t_idx, tree in enumerate(self._trees):
                    predictions_without_this = np.delete(self._predictions, t_idx, axis=0)
                    mean_preds_others = np.mean(predictions_without_this, axis=0)
                    # We revert-check using the entire dataset or just the batch
                    # Doing entire dataset is more accurate
                    reverts = tree.revert_checks(X, y, mean_preds_others, alpha=self.n_trees-1, ccp_alpha = self.ccp_alpha)
                    if reverts > 0:
                        # If we reverted, we need to update the predictions
                        self._predictions[t_idx] = tree.predict(X)
                        self.reversions += reverts
        
        return self 
        

    def predict(self, X_predict):
        X_predict = check_array(X_predict, accept_sparse=False)
        check_is_fitted(self, 'n_features_in_')  # raises NotFittedError if missing
        print(f"This forest has a total of {sum([tree.num_splits for tree in self._trees])} splits")
        print(f"This forest has a total of {len(self._trees)} trees")
        print(f"This forest has reverted {self.reversions} splits")
        
        """We want to go through each tree and combine predictions"""
        predictions = []
        for tree in self._trees:
            predictions.append(tree.predict(X_predict))
        return np.mean(predictions, axis = 0)
