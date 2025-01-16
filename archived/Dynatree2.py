import numpy as np
from Tree2 import Tree
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import r2_score
from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils.validation import check_X_y, check_array


class Dynatree(RegressorMixin, BaseEstimator):
    def __init__(self, n_trees = 10, window = 4, max_depth = 3, min_samples = 2):
        self.n_trees = n_trees
        
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
    
    def add_new_tree(self, tree):
        """Function that deals with adding a new tree to the list of trees, when necessary. """
        self.trees.append(tree)
        self.tree_window.append(tree)
        self.predictions_window.append(tree.get_training_predictions())
        
        #Ensures that the window size remains constant!
        if len(self.tree_window) > self.window:
            self.predictions_window.pop(0)
            self.tree_window.pop(0)
    
    def fit(self, X, y):
        X, y = check_X_y(X, y, accept_sparse=False)
        
        self.trees = []
        self.tree_window = []  #List of trees we are currently looking at
        self.predictions_window = []
        
        X = np.array(X)
        y = np.array(y)
        
        #Creating first tree
        initial_tree = Tree(X, y)
        initial_tree.get_best_split(X, y, np.zeros(len(y)), 0)
        initial_tree.split(X, y)
        self.add_new_tree(initial_tree)
        
                
        while len(self.trees) <= self.n_trees:
            """The key part of the function. This creates splits one by one until all trees are filled"""
            #First: see if I can do better given the trees we already have in the tree window
            error_reductions = []
            num_total_predictions = len(self.predictions_window)
            for (idx, tree) in enumerate(self.tree_window):
                print("Idx is ", idx)
                predictions_without_tree = np.delete(self.predictions_window, idx, 0) #Getting predictions without the tree
                if len(predictions_without_tree) == 0:
                    mean_predictions_without_tree = np.zeros(len(y)) #No other predictions
                else:
                    mean_predictions_without_tree = np.mean(predictions_without_tree, axis = 0)
                """Goes through every tree and spits out which leaf and which splitting function is best, and how much it reduces the error"""
                error_reduction = tree.get_best_split(X, y, mean_predictions_without_tree, num_total_predictions - 1) #Finding the best split
                error_reductions.append(error_reduction)
            
            #Then: see error reduced if we just create a new stump
            new_tree = Tree(X, y, max_depth=self.max_depth, min_samples = self.min_samples)
            mean_predictions = np.mean(self.predictions_window, axis = 0)
            error_reduction = new_tree.get_best_split(X, y, mean_predictions, num_total_predictions)
            
            if error_reduction <= 0 and max(error_reductions) <= 0:
                print("No more error reduction possible")
                break
            elif error_reduction >= max(error_reductions):
                print(f"Creating new tree, error reduction: {error_reduction}")
                new_tree.split(X, y)
                self.add_new_tree(new_tree)
            else:
                print(f"Splitting tree, error reduction: {max(error_reductions)}")
                best_error_idx = np.argmax(error_reductions)
                best_tree = self.tree_window[best_error_idx]
                best_tree.split(X, y)
                self.predictions_window[best_error_idx] = best_tree.get_training_predictions()
            
            print(f"SSE: {np.sum((y - self.predict(X))**2)}")
            
        if len(self.trees) == self.n_trees + 1:
            #We go until the n+1th tree is created, we just don't return it
            self.trees = self.trees[:-1]
        return self 
        

    def predict(self, X_predict):
        X_predict = check_array(X_predict, accept_sparse=False)
        if not hasattr(self, 'trees') or len(self.trees) == 0:
            # If somehow called predict without fit
            return np.zeros(X_predict.shape[0])
        
        """We want to go through each tree and combine predictions"""
        predictions = []
        for tree in self.trees:
            predictions.append(tree.predict(X_predict))
        return np.mean(predictions, axis = 0)
    