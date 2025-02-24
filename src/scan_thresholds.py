import numpy as np
import warnings


def find_split(X, y, other_predictions, leaf_weight, tree_weight, features_to_consider, loss_fn, 
                             min_samples = 2, eta = 0.1, reg_lambda = 0, initial_weight = 'parent', eps = 1e-6):
    """A generic second-order-split approach a la GBM, 
    for any twice-differentiable 'loss_fn'.

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
    
    if len(X) < 2 * min_samples:
        return (-np.inf, None, None, None, None)
    
    best_gain = -np.inf
    curr_best_col = None
    curr_best_splitting_val = None
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
     
    G = np.sum(g_i)
    H = np.sum(h_i) + eps
    prev_score = 1 / 2 * (tree_weight * G + init * reg_lambda) ** 2 / (H * tree_weight**2 + reg_lambda)
    for col in features_to_consider:
        sort_idx = np.argsort(X[:, col])
        X_col_sorted = X[:, col][sort_idx]
        
        #We only consider splits with more than min_samples on each side, after accounting for uniqueness
        X_unique_sorted = np.unique(X_col_sorted)
        
        g_i_sorted = g_i[sort_idx]
        h_i_sorted = h_i[sort_idx]
                
        #We only want to consider the splits when y is at the end of an X value, not in the middle
        #We also care about unique values, so we need to find the last unique index -- this does both
        #Min samples 2, [1, 1, 1, 1, 2, 2, 3, 3] -> [3, 5]
        placements = np.searchsorted(X_col_sorted, X_unique_sorted, side='right') - 1
        placements_correct = [placement for placement in placements if placement >= (min_samples - 1) and placement < len(X_col_sorted) - min_samples]
        
        if len(placements_correct) == 0:
                # No valid splits for this column given the min_samples restriction.
                continue
            
        #Prefix/suffix sums for gradient/hessian
        
        G_L = np.cumsum(g_i_sorted)
        H_L = np.cumsum(h_i_sorted) + eps
        G_R = G_L[-1] - G_L
        H_R = H_L[-1] - H_L + eps
        
        with warnings.catch_warnings():
            """For optimization purposes, we apply the valid mask afterwards, but this means we have some division by 0.
            We use warnings to catch these."""
            warnings.simplefilter("ignore", category=RuntimeWarning)
            left_score = 1 / 2 * (tree_weight * G_L + init * reg_lambda) ** 2 / (H_L * tree_weight**2 + reg_lambda)
            right_score = 1 / 2 * (tree_weight * G_R + init * reg_lambda) ** 2 / (H_R * tree_weight**2 + reg_lambda)
        
        #Gain is NEGATIVE loss
        gain = left_score + right_score - prev_score
        best_idx = placements_correct[np.argmax(gain[placements_correct])]
        
        if gain[best_idx] > best_gain:
            best_gain = gain[best_idx]
            curr_best_col = col
            curr_best_splitting_val = X_col_sorted[best_idx]
            left_delta = - (tree_weight * G_L[best_idx] + init * reg_lambda) / (tree_weight**2 * H_L[best_idx] + reg_lambda)
            right_delta = - (tree_weight * G_R[best_idx] + init * reg_lambda) / (tree_weight**2 * H_R[best_idx] + reg_lambda)
            
            """If we're using argmin, we optimize starting with the argmin of the left and right sides. 
            Otherwise we use the parent weight for the init vals"""
            if initial_weight == 'argmin':
                y_sorted = y[sort_idx]
                left_init = loss_fn.argmin(y_sorted[:best_idx + 1])
                right_init = loss_fn.argmin(y_sorted[best_idx + 1:])
            else:
                left_init = init
                right_init = init
            
            left_val = left_init + eta * left_delta
            right_val = right_init + eta * right_delta
            
    #Calculate actual gain, then return
    if curr_best_col is None:
        return (-np.inf, None, None, None, None)
    
    #Calculate actual gain, then return
    error_before = loss_fn(y, previous_predictions)
    left_error = loss_fn(y[X[:, curr_best_col] <= curr_best_splitting_val], (other_predictions[X[:, curr_best_col] <= curr_best_splitting_val] + left_val * tree_weight))
    right_error = loss_fn(y[X[:, curr_best_col] > curr_best_splitting_val], (other_predictions[X[:, curr_best_col] > curr_best_splitting_val] + right_val * tree_weight))
    error_after = left_error + right_error
    actual_gain = error_before - error_after
    
    return actual_gain, curr_best_col, curr_best_splitting_val, left_val, right_val
    

def get_best_sse(X, y, other_predictions, num_cols_other_prediction, features_to_consider, min_samples = 2, eta = 0.1):
    if len(X) < 2 * min_samples:
        return (np.inf, None, None, None, None) #No error reduction possible
    
    alpha = num_cols_other_prediction #naming easier, we'll be using it a lot
    best_sse = np.inf
    curr_best_col = None
    curr_best_splitting_val = None
    left_val = None
    right_val = None
    
    for col in features_to_consider: #Going through each column
        sort_idx = np.argsort(X[:, col])
        X_col_sorted = X[:, col][sort_idx]
        y_sorted = y[sort_idx]
        other_predictions_sorted = other_predictions[sort_idx]
        
        #We only consider splits with more than min_samples on each side, after accounting for uniqueness
        X_unique_sorted = np.unique(X_col_sorted)
        
        #We only want to consider the splits when y is at the end of an X value, not in the middle
        #We also care about unique values, so we need to find the last unique index -- this does both
        #Min samples 2, [1, 1, 1, 1, 2, 2, 3, 3] -> [3, 5]
        placements = np.searchsorted(X_col_sorted, X_unique_sorted, side='right') - 1
        placements_correct = [placement for placement in placements if placement >= (min_samples - 1) and placement < len(X_col_sorted) - min_samples]

        if len(placements_correct) == 0:
            # No valid splits for this column given the min_samples restriction.
            continue
        
        #Early calculations for prefix/suffix sums, extremely useful for gradient calcs
        #Derivations can be found in cumsum.pdf
        A_i = y_sorted - alpha / (alpha + 1) * other_predictions_sorted
        A_n_prefix = np.cumsum(A_i)
        A_n_prefix_squared = np.cumsum(A_i**2)
        B_n_prefix = ((1 + eta * alpha) * np.cumsum(y_sorted) - eta * alpha * np.cumsum(other_predictions_sorted)) / (1 + alpha)
        prefix_idxs = np.arange(1, len(y_sorted) + 1)
        
        A_n_suffix = A_n_prefix[-1] - A_n_prefix
        A_n_suffix_squared = A_n_prefix_squared[-1] - A_n_prefix_squared
        B_n_suffix = B_n_prefix[-1] - B_n_prefix
        suffix_idxs = prefix_idxs[-1] - prefix_idxs
        
        with warnings.catch_warnings():
            """For optimization purposes, we apply the valid mask afterwards, but this means we have some division by 0.
            We use warnings to catch these."""
            warnings.simplefilter("ignore", category=RuntimeWarning)
            left_sse = A_n_prefix_squared - 2 * B_n_prefix / prefix_idxs * A_n_prefix + B_n_prefix**2 / prefix_idxs
            right_sse = A_n_suffix_squared - 2 * B_n_suffix / suffix_idxs * A_n_suffix + B_n_suffix**2 / suffix_idxs
        
        sse = left_sse + right_sse
                       
        best_idx = placements_correct[np.argmin(sse[placements_correct])]
        if sse[best_idx] < best_sse:
            best_sse = sse[best_idx]
            curr_best_col = col
            curr_best_splitting_val = X_col_sorted[best_idx]
            "Need to figure out how to calculate left_val and right_val, then return them"
            left_idx = X_col_sorted <= curr_best_splitting_val
            right_idx = ~left_idx
            
            left_y, right_y = y_sorted[left_idx], y_sorted[right_idx]
            left_other_predictions, right_other_predictions = other_predictions_sorted[left_idx], other_predictions_sorted[right_idx]
            
            left_val = (1 + eta * alpha) * np.mean(left_y) - eta * alpha * np.mean(left_other_predictions)
            right_val = (1 + eta * alpha) * np.mean(right_y) - eta * alpha * np.mean(right_other_predictions)
    
    return best_sse, curr_best_col, curr_best_splitting_val, left_val, right_val

