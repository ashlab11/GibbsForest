import numpy as np
import random
from line_profiler import profile
from .Tree import Tree
from .Losses import *
from .LeafOrNode import LeafOrNode
from .ParamErrors import check_params
from .hist_splitting import HistSplitter
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

class GibbsForest(RegressorMixin, BaseEstimator):
    def __init__(self, 
                loss_fn = LeastSquaresLoss(),
                n_trees = 100, 
                max_depth = 3,
                max_leaves = np.inf,
                min_samples = 2, 
                feature_subsample_rf = 'sqrt',
                row_subsample_rf = 1,
                feature_subsample_g = 1,
                row_subsample_g = 1,
                warmup_depth = 1,
                leaf_eta = 0.01,
                tree_eta = 0.05,
                reg_lambda = 0, 
                reg_gamma = 0, 
                initial_weight = 'parent', 
                dropout = 0,
                n_bins = 256,
                random_state = None):
        """Parameters:
        loss_fn: loss function to use for the trees (default is LeastSquaresLoss)
        n_trees: number of trees to use in the forest (default is 10)
        max_depth: maximum depth of the trees (default is 3)
        min_samples: minimum number of samples in a node to split (default is 2)
        feature_subsample_rf / feature_subsample_g: fraction of features to consider for each split for the initial RF creation / for later gibbs updates (default is 1)
        row_subsample_rf / row_subsample_g : fraction of rows to consider for each split for the initial RF creation / for later gibbs updates (default is 1)
        warmup_depth: depth of each initial tree before using gibbs algorithm (default is 1)
        leaf_eta: learning rate for leaf weights (default is 0.01)
        tree_eta: learning rate for tree weights (default is 0.05)
        reg_lambda: L2 regularization parameter (default is 0)
        TODO: reg_alpha: L1 regularization parameter (default is 0)
        reg_gamma: min loss reduction to make a split (default is 0)
        initial_weight: initial weight of the tree (default is 'parent')
        dropout: probability of skipping a tree update (default is 0)
        """
        self.loss_fn = loss_fn
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.max_leaves = max_leaves
        self.min_samples = min_samples
        self.feature_subsample_rf = feature_subsample_rf
        self.row_subsample_rf = row_subsample_rf
        self.feature_subsample_g = feature_subsample_g
        self.row_subsample_g = row_subsample_g
        self.warmup_depth = warmup_depth
        self.leaf_eta = leaf_eta
        self.tree_eta = tree_eta
        self.reg_lambda = reg_lambda
        self.reg_gamma = reg_gamma
        self.initial_weight = initial_weight
        self.dropout = dropout
        self.random_state = random_state
        self.n_bins = n_bins
    
    @profile
    def fit(self, X, y):
        X, y = check_X_y(X, y, accept_sparse=False)
        self.n_features_in_ = X.shape[1]  # for sklearn compliance
        
        #Checking parameters -- really long function lol
        (self.loss_fn, self.n_trees, self.max_depth, self.min_samples, self.feature_subsample_rf, self.row_subsample_rf, 
         self.feature_subsample_g, self.row_subsample_g, self.warmup_depth, self.leaf_eta, self.tree_eta, self.reg_lambda, 
         self.reg_gamma, self.initial_weight, self.dropout) = check_params(X, self.loss_fn, self.n_trees, self.max_depth, self.min_samples, self.feature_subsample_rf, self.row_subsample_rf, 
         self.feature_subsample_g, self.row_subsample_g, self.warmup_depth, self.leaf_eta, self.tree_eta, self.reg_lambda, 
         self.reg_gamma, self.initial_weight, self.dropout)
         
        #TODO: Implement native support for categorical variables and EFB
        
        #Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)
                 
        self.weights = np.ones(self.n_trees) * 1 / self.n_trees #Initial weights
        self.feature_importances_ = np.zeros(self.n_features_in_)
        self.feature_splits = np.zeros(self.n_features_in_)
        
        self._trees = []
        self._predictions = np.empty((self.n_trees, len(y)))
        self.hist_splitter = HistSplitter(X, y, n_bins=self.n_bins) #Hist splitter saves X and utilizes row indices for bin calculations
        
        #---- INITIAL TREE CREATION, UP TO WARMUP DEPTH ----#     
        #Creating hist splitter class        
        num_rows_considered = max(int(self.row_subsample_rf * len(y)), 1)
        num_features_considered = max(int(self.n_features_in_ * self.feature_subsample_rf), 1)
        for i in range(self.n_trees):    
            rng = np.random.default_rng(seed = None if self.random_state is None else self.random_state + i)
            bootstrapped_idx = rng.choice(len(X), num_rows_considered, replace = True)
            init = self.loss_fn.argmin(y[bootstrapped_idx])
            
            tree = Tree(init, max_depth=self.max_depth, min_samples = self.min_samples, initial_weight = self.initial_weight, 
                        loss_fn=self.loss_fn, hist_splitter = self.hist_splitter)
            
            #Initial split -- no current tree-level predictions, and no predictions from any other splits
            tree.initial_splits(bootstrapped_idx, self.warmup_depth, num_features_considered = num_features_considered, 
                                rng = rng)
            self._trees.append(tree)
            
            #TODO: implement feature importance here
            self._predictions[i] = tree.predict(X)
            
        #---- GIBBS TREE CREATION ----#
        num_features_considered = max(int(self.n_features_in_ * self.feature_subsample_g), 1)
        num_rows_considered = max(int(self.row_subsample_g * len(y)), 1)
        #Round-robin -- with max_depth = N and initial depth D, we should have 2^N - 2^D splits
        
        for idx in range(min(10000, 2**self.max_depth - 2**self.warmup_depth)):
            num_leaves_per_tree = [tree.num_leaves for tree in self._trees]
            if np.all(np.array(num_leaves_per_tree) >= self.max_leaves):
                break #All trees have reached max leaves
             
            rng = np.random.default_rng(seed = None if self.random_state is None else self.random_state + idx)
            features_considered = rng.choice(self.n_features_in_, num_features_considered)
            rows_considered = rng.choice(len(X), num_rows_considered, replace = False)   
            current_predictions = self.weights @ self._predictions
            #Going through each tree
            for tree_idx, tree in enumerate(self._trees):
                if num_leaves_per_tree[tree_idx] >= self.max_leaves:
                    continue
                
                rng = np.random.default_rng(seed = None if self.random_state is None else self.random_state + tree_idx)
                # ---- Dropout condition: skip updating this tree with probability self.dropout ----
                if rng.random() < self.dropout:
                    continue  # skip update, move to the next tree
                
                #Getting predictions/weights without chosen tree
                predictions_without_tree = current_predictions - self.weights[tree_idx] * self._predictions[tree_idx] 
                
                #Get best split with these weights
                error_reduction, best_split = tree.get_best_split(rows_considered, predictions_without_tree, features_considered, self.weights[tree_idx], 
                                                                  self.leaf_eta)
                if error_reduction > self.reg_gamma: #Only split if gain is above gamma
                    tree.split()
                    
                    #Predictions are based on the entire X, not X_batch, and new_predictions must be updated similarly
                    predictions = tree.predict(X)
                    self._predictions[tree_idx] = predictions
                    current_predictions = predictions_without_tree + self.weights[tree_idx] * predictions
                    
                    #Updating feature importances and splits
                    self.feature_importances_[best_split] += error_reduction
                    self.feature_splits[best_split] += 1
                        
            if idx < 2: #Don't change tree weights for a little while
                continue
            
            #--- UPDATING TREE WEIGHTS ---#            
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
        return self 
        

    def predict(self, X_predict):
        X_predict = check_array(X_predict, accept_sparse=False)
        check_is_fitted(self, 'n_features_in_')  # raises NotFittedError if missing
        
        """We want to go through each tree and combine predictions"""
        predictions = []
        for tree in self._trees:
            predictions.append(tree.predict(X_predict))
        return self.weights @ np.array(predictions)
