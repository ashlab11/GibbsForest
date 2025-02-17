import numpy as np
from .scan_thresholds import get_best_sse, find_split
from .Losses import *
import random

class LeafOrNode:
    def __init__(self, val, curr_depth = 0, max_depth = 3, min_samples = 2, eta = 0.1, initial_weight = 'parent', 
                 loss_fn = LeastSquaresLoss()):
        #Values given from the tree
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.curr_depth = curr_depth
        self.val = val
        self.eta = eta
        self.initial_weight = initial_weight
        self.loss_fn = loss_fn

        #Values calculated by splits
        self.left = None
        self.right = None
        self.curr_best_col = None
        self.curr_best_splitting_val = None
        self.curr_best_error_reduction = None
        self.next_to_split = None
     
         
    def get_best_split(self, X, y, other_predictions, num_cols_other_prediction, features_to_consider):
        """Function that gets the best split for a node or leaf, but doesn't split it yet
        Note: new_tree is a boolean that tells us whether we are splitting a new tree or not, 
        which is necessary for error calculations. 
        Returns the best error reduction and the column to split on"""
        if self.curr_depth == self.max_depth or len(y) <= self.min_samples:
            return (0, -1) #No error reduction possible
        if self.left == None and self.right == None:
            return self.get_best_split_leaf(X, y, other_predictions, num_cols_other_prediction, features_to_consider)
        else:
            return self.get_best_split_node(X, y, other_predictions, num_cols_other_prediction, features_to_consider)
        
    def get_best_split_node(self, X, y, other_predictions, num_cols_other_prediction, features_to_consider):
        """Function that gets the best split for a node, but doesn't split it yet"""
        left_indices = X[:, self.curr_best_col] <= self.curr_best_splitting_val
        right_indices = ~left_indices
        
        left_X = X[left_indices]
        right_X = X[right_indices]

        left_other_predictions = other_predictions[left_indices]
        right_other_predictions = other_predictions[right_indices]
        left_y = y[left_indices]
        right_y = y[right_indices]
                
        left_best_error_reduction, left_col_split = self.left.get_best_split(left_X, left_y, left_other_predictions, num_cols_other_prediction, features_to_consider)
        right_best_error_reduction, right_col_split = self.right.get_best_split(right_X, right_y, right_other_predictions, num_cols_other_prediction, features_to_consider)
        
        if left_best_error_reduction > right_best_error_reduction:
            self.curr_best_error_reduction = left_best_error_reduction
            self.next_to_split = self.left
            return left_best_error_reduction, left_col_split
        else:
            self.curr_best_error_reduction = right_best_error_reduction
            self.next_to_split = self.right
            return right_best_error_reduction, right_col_split

    def get_best_split_leaf(self, X, y, other_predictions, num_cols_other_prediction, features_to_consider):
        """Function that gets the best split for a leaf, but doesn't split it yet."""
        if np.isnan(other_predictions).any():
            print("NAN in other_predictions")
        
        gain, col, splitting_val, left_val, right_val = find_split(
            X, y, other_predictions, self.val, num_cols_other_prediction, features_to_consider = features_to_consider, min_samples=self.min_samples,
            eta = self.eta, loss_fn=self.loss_fn, initial_weight = self.initial_weight)
    
        self.curr_best_col = col
        self.curr_best_splitting_val = splitting_val
        self.left_val = left_val
        self.right_val = right_val
            
        return gain, self.curr_best_col
    
    def initial_split(self, X, y, warmup_depth, features_considered):
        """Function that does the initial splits, up to a warmup depth
        NOTE: Setting initial weight = parent and eta = 1 is equivalent to using argmin with 0 eta for initial splits.
        I chose to use the former for theoretical simplicity, but the latter is equally efficient.
        """
        if self.curr_depth < warmup_depth:
            #Testing parent 1 vs argmin 0
            #On MSE, parent 1 and argmin 0 are the same
            gain, col, splitting_val, left_val, right_val = find_split(X, y, np.zeros(len(y)), self.val, 0, features_to_consider = features_considered, min_samples=self.min_samples,
                    eta = 1, loss_fn=self.loss_fn, initial_weight = "parent")
                        
            if splitting_val is None:
                #When no further splits can be made, stop initial splits
                return
            
            self.curr_best_col = col
            self.curr_best_splitting_val = splitting_val
            self.left_val = left_val
            self.right_val = right_val
            self.split(X, y)
            
            left_indices = X[:, self.curr_best_col] <= self.curr_best_splitting_val
            right_indices = ~left_indices
            self.left.initial_split(X[left_indices], y[left_indices], warmup_depth, features_considered)
            self.right.initial_split(X[right_indices], y[right_indices], warmup_depth, features_considered) 
    
    def split(self, X, y):
        """Function that ACTUALLY splits the node, given information we created when testing splits"""
        if self.curr_best_error_reduction == 0:
            #Should never happen, since we just don't split in the first place
            raise ValueError("Error: No error reduction possible")
        else:
            if self.next_to_split == None:
                self.split_leaf(X, y)
            else:
                #Splitting the node for real!
                if self.next_to_split == self.left:
                    left_idx = X[:, self.curr_best_col] <= self.curr_best_splitting_val
                    left_X = X[left_idx]
                    left_y = y[left_idx]
                    self.left.split(left_X, left_y)
                else:
                    right_idx = X[:, self.curr_best_col] > self.curr_best_splitting_val
                    right_X = X[right_idx]
                    right_y = y[right_idx]
                    self.right.split(right_X, right_y)
                
    def split_leaf(self, X, y):
        self.left = LeafOrNode(self.left_val, curr_depth= self.curr_depth + 1, max_depth = self.max_depth, min_samples = self.min_samples, eta = self.eta, initial_weight = self.initial_weight)
        self.right = LeafOrNode(self.right_val, curr_depth = self.curr_depth + 1, max_depth = self.max_depth, min_samples = self.min_samples, eta = self.eta, initial_weight = self.initial_weight)
        return None
     
    def predict(self, X_predict = None):
        """Function that predicts the values for a given X_predict. If conducted in training mode, we can set new predictions"""
        if self.left == None and self.right == None:
            return np.ones(len(X_predict)) * self.val
        else:
            left_indices = X_predict[:, self.curr_best_col] <= self.curr_best_splitting_val
            right_indices = ~left_indices
            
            left_X = X_predict[left_indices]
            right_X = X_predict[right_indices]
            
            left_predictions = self.left.predict(left_X)
            right_predictions = self.right.predict(right_X)
            
            predictions = np.zeros(len(X_predict))
            predictions[left_indices] = left_predictions
            predictions[right_indices] = right_predictions
    
            return predictions
        
    def __repr__(self):
        if self.left == None and self.right == None:
            return f"Leaf with value {self.val}"
        return f"Node with splitting value {self.curr_best_splitting_val} and column {self.curr_best_col}"
