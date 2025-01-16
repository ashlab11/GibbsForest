import numpy as np

class LeafOrNode:
    def __init__(self, y, min_samples = 1):
        self.left = None
        self.right = None
        self.min_samples = min_samples
        
        #Here, we get the value of the leaf.
        #I will implement gradient descent on this later
        self.val = np.mean(y)
        
        self.curr_best_col = None
        self.curr_best_splitting_val = None
        self.curr_best_error_reduction = None
        self.next_to_split = None
                
    def get_best_split(self, X, y, tree_level_predictions, other_predictions, num_cols_other_prediction, new_tree = True):
        if self.left == None and self.right == None:
            return self.get_best_split_leaf(X, y, tree_level_predictions, other_predictions, num_cols_other_prediction, new_tree)
        else:
            return self.get_best_split_node(X, y, tree_level_predictions, other_predictions, num_cols_other_prediction)
        
    def get_best_split_node(self, X, y, tree_level_predictions, other_predictions, num_cols_other_prediction):
        """Function that gets the best split for a node, but doesn't split it yet"""
        left_indices = X[:, self.curr_best_col] <= self.curr_best_splitting_val
        right_indices = X[:, self.curr_best_col] > self.curr_best_splitting_val
        
        left_X = X[left_indices]
        right_X = X[right_indices]
        left_tree_level_predictions = tree_level_predictions[left_indices]
        right_tree_level_predictions = tree_level_predictions[right_indices]
        
        left_other_predictions = other_predictions[left_indices]
        right_other_predictions = other_predictions[right_indices]
        left_y = y[left_indices]
        right_y = y[right_indices]
        
        left_best_error_reduction = self.left.get_best_split(left_X, left_y, left_tree_level_predictions, left_other_predictions, num_cols_other_prediction, 
                                                             new_tree=False)
        right_best_error_reduction = self.right.get_best_split(right_X, right_y, right_tree_level_predictions, right_other_predictions, num_cols_other_prediction, 
                                                               new_tree =False)
        
        if left_best_error_reduction > right_best_error_reduction:
            self.curr_best_error_reduction = left_best_error_reduction
            self.next_to_split = self.left
            return left_best_error_reduction
        else:
            self.curr_best_error_reduction = right_best_error_reduction
            self.next_to_split = self.right
            return right_best_error_reduction
        
    def get_best_split_leaf(self, X, y, tree_level_predictions, other_predictions, num_cols_other_prediction, new_tree):
        """Function that gets the best split for a leaf, but doesn't split it yet. 
        TBD: Implement gradient descent here
        To fix the problem of many trees, we will add in a new_tree parameter"""
        curr_best_error_reduction = -np.inf
        if new_tree:
            old_predictions = other_predictions
        else:
            old_predictions = (tree_level_predictions + other_predictions * num_cols_other_prediction) / (num_cols_other_prediction + 1)
        total_error = np.sum((y - old_predictions)**2) #Getting total error that we want to reduce
        
        for col in range(len(X[0])): #Going through each column
            unique_vals_sorted = np.unique(X[:, col])
            for val in unique_vals_sorted: #Going through each unique value, and getting predictions based on means of split
                below_indices = X[:, col] <= val
                above_indices = X[:, col] > val
                
                if np.sum(below_indices) < self.min_samples or np.sum(above_indices) < self.min_samples:
                    continue
                
                #GD here
                below_mean, above_mean = np.mean(y[below_indices]), np.mean(y[above_indices])
                
                #Combining predictions from other trees
                below_predictions = (other_predictions[below_indices] * num_cols_other_prediction + below_mean) / (num_cols_other_prediction + 1)
                above_predictions = (other_predictions[above_indices] * num_cols_other_prediction + above_mean) / (num_cols_other_prediction + 1)
                below_error = np.sum((y[X[:, col] <= val] - below_predictions)**2)
                above_error = np.sum((y[X[:, col] > val] - above_predictions)**2)
                
                error_reduction = total_error - below_error - above_error
                
                #Update best error reduction
                if error_reduction > curr_best_error_reduction:
                    curr_best_error_reduction = error_reduction
                    self.curr_best_col = col
                    self.curr_best_splitting_val = val
                    
        self.curr_best_error_reduction = curr_best_error_reduction
        
        return curr_best_error_reduction
        
    def split(self, X, y):
        """Function that ACTUALLY splits the node, given information we created when testing splits"""
        if self.curr_best_error_reduction == 0:
            #Should never happen, since we just don't split in the first place
            raise ("Error: No error reduction possible")
        else:
            if self.next_to_split == None:
                self.split_leaf(X, y)
            else:
                #Splitting the node for real!
                if self.next_to_split == self.left:
                    left_X = X[X[:, self.curr_best_col] <= self.curr_best_splitting_val]
                    left_y = y[X[:, self.curr_best_col] <= self.curr_best_splitting_val]
                    self.left.split(left_X, left_y)
                else:
                    right_X = X[X[:, self.curr_best_col] > self.curr_best_splitting_val]
                    right_y = y[X[:, self.curr_best_col] > self.curr_best_splitting_val]
                    self.right.split(right_X, right_y)
                
    def split_leaf(self, X, y):  
        below_y = y[X[:, self.curr_best_col] <= self.curr_best_splitting_val]
        above_y = y[X[:, self.curr_best_col] > self.curr_best_splitting_val]
        self.left = LeafOrNode(below_y, min_samples = self.min_samples)
        self.right = LeafOrNode(above_y, min_samples = self.min_samples)
        return None
     
    def predict(self, X_predict = None):
        if self.left == None and self.right == None:
            return np.ones(len(X_predict)) * self.val
        else:
            left_indices = X_predict[:, self.curr_best_col] <= self.curr_best_splitting_val
            right_indices = X_predict[:, self.curr_best_col] > self.curr_best_splitting_val
            
            left_X = X_predict[left_indices]
            right_X = X_predict[right_indices]
            
            left_predictions = self.left.predict(left_X)
            right_predictions = self.right.predict(right_X)
            
            predictions = np.zeros(len(X_predict))
            predictions[left_indices] = left_predictions
            predictions[right_indices] = right_predictions
            
            return predictions
        
class Tree:
    def __init__(self, X, y, min_samples = 1, max_depth = 3):
        self.root = LeafOrNode(y, min_samples = min_samples)
        self.min_samples = min_samples
        self.max_depth = max_depth
        self.predictions = self.root.predict(X)
    def get_best_split(self, X, y, other_predictions, num_cols_other_prediction):
        return self.root.get_best_split(X,y, self.predictions, other_predictions, num_cols_other_prediction) 
    def split(self, X, y):
        """Function that splits the tree, keeping predictions up to date"""
        self.root.split(X, y)
        self.predictions = self.root.predict(X)
    def get_training_predictions(self):
        """Function that gets the training predictions"""
        return self.predictions
    def predict(self, X_predict):
        return self.root.predict(X_predict)

