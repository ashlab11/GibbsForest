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
                 feature_subsample = 1, row_subsample = 1, warmup_depth = 1, eta = 0.01, tree_eta = 0.05,
                 reg_lambda = 0, reg_gamma = 0, initial_weight = 'parent', 
                 eta_decay = 1, dropout = 0):
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
        self.tree_eta = tree_eta
    
    
    def fit(self, X, y):
        X, y = check_X_y(X, y, accept_sparse=False)
        self.n_features_in_ = X.shape[1]  # for sklearn compliance
        
        #Checking parameters
        self.loss_fn, self.n_trees, self.max_depth, self.min_samples, self.feature_subsample, self.row_subsample, self.warmup_depth, self.eta, self.reg_lambda, self.reg_gamma, self.initial_weight, self.eta_decay, self.dropout = check_params(
            self.loss_fn, self.n_trees, self.max_depth, self.min_samples, self.feature_subsample, self.row_subsample, self.warmup_depth, self.eta, self.reg_gamma, self.reg_lambda, self.initial_weight, self.eta_decay, self.dropout)
        
        self.weights = np.ones(self.n_trees) * 1 / self.n_trees #Initial weights
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

        #---- INITIAL TREE CREATION, UP TO WARMUP DEPTH ----#     
        for i in range(self.n_trees):
            bootstrapped_idx = np.random.choice(len(X), self.num_rows_considering, replace = True)
            bootstrapped_X = X[bootstrapped_idx]
            bootstrapped_y = y[bootstrapped_idx]
            
            tree = Tree(bootstrapped_X, bootstrapped_y, num_features_considering = self.num_features_considering, 
                        max_depth=self.max_depth, min_samples = self.min_samples, initial_weight = self.initial_weight, 
                        loss_fn=self.loss_fn)
            #Initial split -- no current tree-level predictions, and no predictions from any other splits
            tree.initial_splits(bootstrapped_X, bootstrapped_y, self.warmup_depth)
            self._trees.append(tree)
            
            #TODO: implement feature importance here
            self._predictions[i] = tree.predict(X)
            
        #---- GIBBS TREE CREATION ----#
        #Round-robin -- with max_depth = N and initial depth D, we should have 2^N - 2^D splits
        for idx in range(2**self.max_depth - 2**self.warmup_depth):
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
                individual_predictions_without_tree = np.delete(batch_predictions, tree_idx, 0)
                weights_without_tree = np.delete(self.weights, tree_idx)
                predictions_without_tree = weights_without_tree @ individual_predictions_without_tree
                
                #Get best split with these weights
                error_reduction, best_split = tree.get_best_split(X_batch, y_batch, predictions_without_tree, self.weights[tree_idx], 
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
            
            if idx < 2: #Don't change tree weights for a little while
                continue
            
            #Updating tree weights with math in adaptive weights.pdf
            current_predictions = self.weights @ self._predictions
            current_error = self.loss_fn(y, current_predictions)
            
            current_predictions = current_predictions
            
            pred_gradients = self.loss_fn.gradient(y, current_predictions)
            pred_gradients = pred_gradients.reshape(-1, 1)
            gradients = self._predictions @ pred_gradients
            
            pred_hessian = self.loss_fn.hessian(y, current_predictions)
            pred_hessian = pred_hessian
            hessian = self._predictions * pred_hessian @ self._predictions.T
            
            damping = 1e-6
            inv_hessian = np.linalg.inv(hessian + damping * np.eye(self.n_trees))
            
            weight_update = - inv_hessian @ (gradients - np.ones((self.n_trees, 1)) * 
                                                (np.ones((1, self.n_trees)) @ inv_hessian @ gradients) / (np.ones((1, self.n_trees)) @ inv_hessian @ np.ones((self.n_trees, 1))))
            weight_update = weight_update.flatten()
            
            self.weights += self.tree_eta * weight_update
            #Updating weights, removing trees below 0
            below_0_idx = np.where(self.weights < 0)
            self.weights = np.delete(self.weights, below_0_idx)
            self._predictions = np.delete(self._predictions, below_0_idx, 0)
            self._trees = np.delete(self._trees, below_0_idx)
            self.weights = self.weights / np.sum(self.weights)
            self.n_trees = len(self.weights)         
            
            errors_after = self.loss_fn(y, self.weights @ self._predictions)
            print(f"Error after: {errors_after}")
            print(f"Error differential: {errors_after - current_error}")   
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
        return self.weights @ np.array(predictions)
