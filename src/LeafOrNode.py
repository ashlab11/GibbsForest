import numpy as np
from .hist_splitting import HistSplitter
from .scan_thresholds import get_best_sse, find_split
from .Losses import *
import time
from line_profiler import profile

class LeafOrNode:
    def __init__(self, val, rows_considered, hist_splitter : HistSplitter, curr_depth = 0, max_depth = 3, min_samples = 2, initial_weight = 'parent', 
                 loss_fn = LeastSquaresLoss()):
        #Values given from the parent or tree
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.curr_depth = curr_depth
        self.val = val
        self.initial_weight = initial_weight
        self.loss_fn = loss_fn
        self.hist_splitter = hist_splitter
        self.rows_considered = rows_considered

        #Values calculated by splits
        self.left = None
        self.right = None
        self.curr_best_col = None
        self.curr_best_splitting_val = None
        self.curr_best_error_reduction = None
        self.next_to_split = None
     
         
    def get_best_split(self, row_idxs, other_predictions, features_considered, tree_weight, eta):
        """Function that gets the best split for a node or leaf, but doesn't split it yet
        Note: new_tree is a boolean that tells us whether we are splitting a new tree or not, 
        which is necessary for error calculations. 
        Returns the best error reduction and the column to split on"""
        if self.curr_depth == self.max_depth or len(row_idxs) <= self.min_samples:
            return (0, -1) #No error reduction possible
        if self.left == None and self.right == None:
            return self.get_best_split_leaf(row_idxs, other_predictions, features_considered, tree_weight, eta)
        else:
            return self.get_best_split_node(row_idxs, other_predictions, features_considered, tree_weight, eta)
    
    @profile
    def get_best_split_node(self, row_idxs, other_predictions, features_considered, tree_weight, eta):
        """Function that gets the best split for a node, but doesn't split it yet"""
        
        left_idxs, right_idxs = self.hist_splitter.split(row_idxs, self.curr_best_col, self.curr_best_splitting_val, self.missing_goes_left)
                
        left_best_error_reduction, left_col_split = self.left.get_best_split(left_idxs, other_predictions, features_considered, tree_weight, eta)
        right_best_error_reduction, right_col_split = self.right.get_best_split(right_idxs, other_predictions, features_considered, tree_weight, eta)
        
        if left_best_error_reduction > right_best_error_reduction:
            self.curr_best_error_reduction = left_best_error_reduction
            self.next_to_split = self.left
            return left_best_error_reduction, left_col_split
        else:
            self.curr_best_error_reduction = right_best_error_reduction
            self.next_to_split = self.right
            return right_best_error_reduction, right_col_split
    @profile
    def get_best_split_leaf(self, row_idxs, other_predictions, features_considered, tree_weight, eta):
        """Function that gets the best split for a leaf, but doesn't split it yet."""
        other_predictions_considered = other_predictions[row_idxs]
        
        gain, col, splitting_val, left_val, right_val, missing_goes_left = self.hist_splitter.find_split_hist(row_idxs, other_predictions_considered, self.val, tree_weight, features_to_consider=features_considered, 
                                                                                            min_samples=self.min_samples, eta = eta, loss_fn=self.loss_fn, 
                                                                                            initial_weight = self.initial_weight)
        
        self.curr_best_col = col
        self.curr_best_splitting_val = splitting_val
        self.left_val = left_val
        self.right_val = right_val
        self.missing_goes_left = missing_goes_left
            
        return gain, self.curr_best_col
    
    def initial_split(self, row_idxs, warmup_depth, features_considered):
        """Function that does the initial splits, up to a warmup depth
        NOTE: Setting initial weight = parent and eta = 1 is equivalent to using argmin with 0 eta for initial splits.
        I chose to use the former for theoretical simplicity, but the latter is equally efficient.
        """
        splits = 0
        left_splits = 0
        right_splits = 0
        if self.curr_depth < warmup_depth:
            gain, col, splitting_val, left_val, right_val, missing_goes_left = self.hist_splitter.find_split_hist(row_idxs, other_predictions= np.zeros_like(row_idxs), leaf_weight = self.val, tree_weight=1, features_to_consider = features_considered, min_samples=self.min_samples,
                    eta = 1, loss_fn=self.loss_fn, initial_weight = "parent")
                        
            if splitting_val is None:
                #When no further splits can be made, stop initial splits
                return 0
            
            self.curr_best_col = col
            self.curr_best_splitting_val = splitting_val
            self.left_val = left_val
            self.right_val = right_val
            self.missing_goes_left = missing_goes_left
            self.split()
            left_idxs, right_idxs = self.hist_splitter.split(row_idxs, col, splitting_val, missing_goes_left)
            
            splits += 1
            
            left_splits = self.left.initial_split(left_idxs, warmup_depth, features_considered)
            right_splits = self.right.initial_split(right_idxs, warmup_depth, features_considered) 
            
        return splits + left_splits + right_splits
    
    def split(self):
        """Function that ACTUALLY splits the node, given information we created when testing splits"""
        if self.curr_best_error_reduction == 0:
            #Should never happen, since we just don't split in the first place
            raise ValueError("Error: No error reduction possible")
        else:
            if self.next_to_split == None:
                #Creating a leaf, keeping idxs for future use
                return self.split_leaf()
            else:
                #Splitting the node for real!
                if self.next_to_split == self.left:
                    return self.left.split()
                else:
                    return self.right.split()
                
    def split_leaf(self):
        left_considered, right_considered = self.hist_splitter.split(self.rows_considered, self.curr_best_col, self.curr_best_splitting_val, self.missing_goes_left)
        
        self.left = LeafOrNode(self.left_val, rows_considered = left_considered, curr_depth= self.curr_depth + 1, max_depth = self.max_depth, min_samples = self.min_samples, initial_weight = self.initial_weight, loss_fn = self.loss_fn, hist_splitter=self.hist_splitter)
        self.right = LeafOrNode(self.right_val, rows_considered = right_considered, curr_depth = self.curr_depth + 1, max_depth = self.max_depth, min_samples = self.min_samples, initial_weight = self.initial_weight, loss_fn = self.loss_fn, hist_splitter=self.hist_splitter)
        return left_considered, right_considered, self.left_val, self.right_val
     
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
