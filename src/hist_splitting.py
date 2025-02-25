import numpy as np
from line_profiler import profile
import warnings
from numba import njit

class HistSplitter:
    """
    Splitting class that utilizes global-esque variables to help in splitting
    
    @params: X: ndarray, shape (n_samples, n_features)
    
    @params: y: ndarray, shape (n_samples,)
    
    @params: n_bins: int, number of bins to use for histogram
            
    Values: 
    
    n_vals: int, number of samples
    
    n_features: int, number of features
    
    n_unique: list of int, number of unique values in each feature
    
    actual_bin_num: list of int, number of bins to use for each feature
    
    bins: list of ndarray, the bins for each feature
    
    bin_indices_for_col: list of ndarray, the bin indices for each feature
    """
    
    def __init__(self, X, y, n_bins = 256):
        """Splitting class that utilizes global-esque variables to help in splitting"""
        self.X = X
        self.y = y
        self.missing_idxs = np.isnan(X)
        self.n_vals = X.shape[0]
        self.n_features = X.shape[1]
        self.n_unique = [len(np.unique(X[:, i])) for i in range(X.shape[1])]
        self.actual_bin_num = np.minimum(self.n_unique, n_bins)
        self.bins = [np.linspace(np.min(X[:, i]), np.max(X[:, i]), self.actual_bin_num[i]) for i in range(X.shape[1])]
        self.bin_indices_for_col = np.array([np.digitize(X[:, i], self.bins[i], right=True) for i in range(X.shape[1])]).T
    
    def split(self, row_idxs, col_idx, split_val, missing_goes_left):
        """Function that splits the data given a column and a splitting value"""
        X = self.X[row_idxs]
        
        if missing_goes_left:
            left_idx = np.isnan(X[:, col_idx]) | (X[:, col_idx] <= split_val)
            right_idx = ~left_idx
        else:
            right_idx = np.isnan(X[:, col_idx]) | (X[:, col_idx] > split_val)
            left_idx = ~right_idx
        
        #We want indices to be numeric, not boolean
        left_idx = row_idxs[left_idx]
        right_idx = row_idxs[right_idx]
        return left_idx, right_idx
    
    @profile
    def find_split_hist_col_sparsity(self, col_idx, row_idx, g_i, h_i, init, tree_weight, reg_lambda,
                            min_samples = 2, eps = 1e-6):
        """Splitting a single column of X, using histogram splitting for computational efficiency"""
        if self.n_unique[col_idx] == 1:
            # Can't split on a single unique value
            return (-np.inf, None, False)
                
        bins = self.bins[col_idx]
                
        # Separate out missing vs non-missing
        missing_mask = self.missing_idxs[row_idx, col_idx] #Boolean mask

        # Values for non-missing
        g_i_non_missing = g_i[~missing_mask]
        h_i_non_missing = h_i[~missing_mask]
        n_missing = np.sum(missing_mask)

        # If everything is missing or everything is non-missing with < 2*min_samples, skip
        if len(g_i_non_missing) < min_samples:
            # It effectively can't split
            return (-np.inf, None, False)
        
        bin_indices = self.bin_indices_for_col[row_idx, col_idx][~missing_mask]
        G = np.bincount(bin_indices, weights=g_i_non_missing)
        H = np.bincount(bin_indices, weights=h_i_non_missing)
        counts = np.bincount(bin_indices)
    
        G_prefix = np.cumsum(G)
        H_prefix = np.cumsum(H) + eps
        counts_prefix = np.cumsum(counts)
        G_suffix = G_prefix[-1] - G_prefix
        H_suffix = H_prefix[-1] - H_prefix + eps
        counts_suffix = counts_prefix[-1] - counts_prefix
        
        best_gain = -np.inf
        best_split = None
        missing_goes_left = True
        
        #--- LOOP OVER BINS IF MISSING DOESN'T EXIST ---#
        if n_missing == 0:
            left_scores = 1 / 2 * (tree_weight * G_prefix + init * reg_lambda) ** 2 / (H_prefix * tree_weight**2 + reg_lambda)
            right_scores = 1 / 2 * (tree_weight * G_suffix + init * reg_lambda) ** 2 / (H_suffix * tree_weight**2 + reg_lambda)
            gains = left_scores + right_scores
            valid_placements = np.where((counts_prefix >= min_samples) & (counts_suffix >= min_samples))[0]
            if len(valid_placements) == 0:
                return (-np.inf, None, False)
            best_idx = valid_placements[np.argmax(gains[valid_placements])]
            if gains[best_idx] > best_gain:
                best_gain = gains[best_idx]
                best_split = bins[best_idx]
            return (best_gain, best_split, missing_goes_left)
        
        #--- LOOP OVER BINS, ONLY IF MISSING EXISTS ---#
        G_missing = np.sum(g_i[missing_mask])
        H_missing = np.sum(h_i[missing_mask])
        
        #--- MISSING GOES LEFT ----
        valid_placements = np.where((counts_prefix + n_missing >= min_samples) & (counts_suffix >= min_samples))[0]
        G_prefix_left = G_prefix + G_missing
        H_prefix_left = H_prefix + H_missing + eps
        left_scores_missing_left = 1 / 2 * (tree_weight * G_prefix_left + init * reg_lambda) ** 2 / (H_prefix_left * tree_weight**2 + reg_lambda)
        right_scores_missing_left = 1 / 2 * (tree_weight * G_suffix + init * reg_lambda) ** 2 / (H_suffix * tree_weight**2 + reg_lambda)
        gains_left = left_scores_missing_left + right_scores_missing_left
        if len(valid_placements) > 0:
            best_idx = valid_placements[np.argmax(gains_left[valid_placements])]
            if gains_left[best_idx] > best_gain:
                best_gain = gains_left[best_idx]
                best_split = bins[best_idx]
                missing_goes_left = True
                
        #--- MISSING GOES RIGHT ----
        valid_placements = np.where((counts_suffix + n_missing >= min_samples) & (counts_prefix >= min_samples))[0]
        G_suffix_right = G_suffix + G_missing
        H_suffix_right = H_suffix + H_missing + eps
        left_scores_missing_right = 1 / 2 * (tree_weight * G_prefix + init * reg_lambda) ** 2 / (H_prefix * tree_weight**2 + reg_lambda)
        right_scores_missing_right = 1 / 2 * (tree_weight * G_suffix_right + init * reg_lambda) ** 2 / (H_suffix_right * tree_weight**2 + reg_lambda)
        gains_right = left_scores_missing_right + right_scores_missing_right
        if len(valid_placements) > 0:
            best_idx = valid_placements[np.argmax(gains_right[valid_placements])]
            if gains_right[best_idx] > best_gain:
                best_gain = gains_right[best_idx]
                best_split = bins[best_idx]
                missing_goes_left = False
                
        return (best_gain, best_split, missing_goes_left)
            
    def find_split_hist(self, row_idxs, other_predictions, leaf_weight, tree_weight, features_to_consider, loss_fn,
                        min_samples = 2, eta = 0.1, reg_lambda = 0, initial_weight = 'parent', eps = 1e-6):
        """A generic second-order-split approach a la GBM, 
        for any twice-differentiable 'loss_fn'. Uses histogram splitting for computational efficiency.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
        y : ndarray, shape (n_samples,)
        other_predictions : ndarray, shape (n_samples,)
        The predictions from the rest of the ensemble (i.e. alpha many).
        parent_leaf_weight : float
            The parent's weight 
        num_cols_other_preds : int
            The alpha (number of other trees in ensemble).
        loss_fn : object 
            Must implement .gradient(target, pred), .hessian(target, pred), and .argmin(target).
        features_to_consider : list of int
            Indices of columns to consider as potential splits.
        min_samples : int
            Minimum #samples required in each child for a valid split.
        eta : float
            Learning rate scaling factor.
        reg_lambda : float
            L2 regularization.
        initial_weight : 'parent' or 'argmin'
            Whether each child is built around the parent's weight or 
            we reset each child to local argmin of that child’s y. 
        eps : float
            Small offset to avoid dividing by zero in Hessians.
        Returns
        -------
        best_gain : float
        best_col : int
        best_split_val : float
        left_child_weight : float
        right_child_weight : float"""

        X = self.X[row_idxs]
        y = self.y[row_idxs]
        
        if len(row_idxs) < 2 * min_samples:
            return (-np.inf, None, None, None, None, True)
        
        best_gain = -np.inf
        curr_best_col = None
        curr_best_splitting_val = None
        missing_goes_left = True
        left_val = None
        right_val = None
        
        
        #Initial weight for the leaf, pre update
        if initial_weight == 'parent':
            init = leaf_weight
        elif initial_weight == 'argmin':
            init = loss_fn.argmin(y)
        else:
            raise ValueError('Initial weight must be parent or argmin')
            
        #Predictions before splitting
        previous_predictions = other_predictions + init * tree_weight
        g_i = loss_fn.gradient(y, previous_predictions)
        h_i = loss_fn.hessian(y, previous_predictions)
        
        # -- MAIN LOOP OVER FEATURES -- #
        for col_idx in features_to_consider:            
            col_gain, col_split, col_missing_goes_left = self.find_split_hist_col_sparsity(col_idx, row_idxs, g_i, h_i, init, tree_weight, reg_lambda,
                            min_samples = min_samples, eps = eps)
            if col_gain > best_gain:
                best_gain = col_gain
                curr_best_col = col_idx
                curr_best_splitting_val = col_split
                missing_goes_left = col_missing_goes_left
        
        #Calculate actual gain, then return
        if curr_best_col is None:
            return (-np.inf, None, None, None, None, True)
        
        if missing_goes_left:
            left_idx = np.isnan(X[:, curr_best_col]) | (X[:, curr_best_col] <= curr_best_splitting_val)
            right_idx = ~left_idx
        else:
            right_idx = np.isnan(X[:, curr_best_col]) | (X[:, curr_best_col] > curr_best_splitting_val)
            left_idx = ~right_idx
        
        left_delta = - (tree_weight * np.sum(g_i[left_idx]) + init * reg_lambda) / (tree_weight**2 * np.sum(h_i[left_idx]) + reg_lambda)
        right_delta = - (tree_weight * np.sum(g_i[right_idx]) + init * reg_lambda) / (tree_weight**2 * np.sum(h_i[right_idx]) + reg_lambda)
        
        """If we're using argmin, we optimize starting with the argmin of the left and right sides. 
                Otherwise we use the parent weight for the init vals"""
        if initial_weight == 'argmin':
            left_init = loss_fn.argmin(y[left_idx])
            right_init = loss_fn.argmin(y[right_idx])
        else:
            left_init = init
            right_init = init
            
        left_val = left_init + eta * left_delta
        right_val = right_init + eta * right_delta
        
        #Calculate actual gain, then return
        error_before = loss_fn(y, previous_predictions)
        left_error = loss_fn(y[left_idx], (other_predictions[left_idx] + left_val * tree_weight))
        right_error = loss_fn(y[right_idx], (other_predictions[right_idx] + right_val * tree_weight))
        error_after = left_error + right_error
        actual_gain = error_before - error_after
        
        return actual_gain, curr_best_col, curr_best_splitting_val, left_val, right_val, missing_goes_left
