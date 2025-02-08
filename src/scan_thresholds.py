import numpy as np
from line_profiler import profile
import warnings


def get_best_sse_arbitrary_slow(X, y, other_predictions, num_cols_other_prediction, features_to_consider, loss_fn, min_samples = 2, delta = 0.1,
                                reg_lambda = 0):
    if len(X) < 2 * min_samples:
        return (np.inf, None, None, None, None)
    
    alpha = num_cols_other_prediction #naming easier, we'll be using it a lot
    best_loss = np.inf
    curr_best_col = None
    curr_best_splitting_val = None
    left_val = None
    right_val = None
        
    for col in features_to_consider:
        for splitting_val in np.unique(X[:, col]):
            left_idx = X[:, col] <= splitting_val
            right_idx = ~left_idx
            
            left_y, right_y = y[left_idx], y[right_idx]
            left_other_predictions, right_other_predictions = other_predictions[left_idx], other_predictions[right_idx]
            
            left_gamma = loss_fn.argmin(left_y)
            right_gamma = loss_fn.argmin(right_y)

            

            left_val = left_gamma - delta * np.mean(left_other_predictions)
            right_val = right_gamma - delta * np.mean(right_other_predictions)
            
            left_predictions = (alpha * left_other_predictions + left_val * np.ones_like(left_y)) / (1 + alpha)
            right_predictions = (alpha * right_other_predictions + right_val * np.ones_like(right_y)) / (1 + alpha)
            
            left_loss = loss_fn.function(left_y, left_predictions)
            right_loss = loss_fn.function(right_y, right_predictions)
            
            initial_predictions = (alpha * other_predictions + left_val * left_idx + right_val * right_idx) / (1 + alpha)
            loss = loss_fn.function(y, initial_predictions)
            
            if loss < best_loss:
                best_loss = loss
                curr_best_col = col
                curr_best_splitting_val = splitting_val
                
                
    return best_loss, curr_best_col, curr_best_splitting_val, left_val, right_val

def get_best_sse_arbitary(X, y, other_predictions, num_cols_other_prediction, features_to_consider, loss_fn, min_samples = 2, delta = 0.1, 
                          reg_lambda = 0):
    """Function that calculates the best SSE for any arbitrary twice-differentiable loss function."""
    if len(X) < 2 * min_samples:
        return (np.inf, None, None, None, None)
    
    alpha = num_cols_other_prediction #naming easier, we'll be using it a lot
    best_gain = -np.inf
    curr_best_col = None
    curr_best_splitting_val = None
    left_val = None
    right_val = None
    
    for col in features_to_consider:
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
                
        #Calculation of gradient/hessian for each split
        global_gamma = loss_fn.argmin(y_sorted)
        initial_predictions = (alpha * other_predictions_sorted + global_gamma) / (1 + alpha)
        g_i = 1 / (alpha + 1) * loss_fn.gradient(y_sorted, initial_predictions)
        h_i = 1 / (alpha + 1)**2 * loss_fn.hessian(y_sorted, initial_predictions)
        
        #Prefix/suffix sums for gradient/hessian
        eps = 1e-6
        g_prefix = np.cumsum(g_i)
        h_prefix = np.cumsum(h_i) + eps 
        g_suffix = g_prefix[-1] - g_prefix
        h_suffix = h_prefix[-1] - h_prefix + eps
        
        #Calculating the approximate gain for each split
        gain = 1 / 2 * (g_prefix**2 / (h_prefix + reg_lambda) + g_suffix**2 / (h_suffix + reg_lambda) - 
                        (g_prefix + g_suffix)**2 / (h_prefix + h_suffix + reg_lambda))
        
        best_idx = placements_correct[np.argmax(gain[placements_correct])]
        best_gain = gain[best_idx]
        curr_best_col = col
        curr_best_splitting_val = X_col_sorted[best_idx]
        
        #Find left and right values, then put them
        left_gamma = loss_fn.argmin(y_sorted[:best_idx + 1])
        right_gamma = loss_fn.argmin(y_sorted[best_idx + 1:])
        
        left_val = left_gamma - delta * g_prefix[best_idx] / (h_prefix[best_idx] + reg_lambda)
        right_val = right_gamma - delta * g_suffix[best_idx] / (h_suffix[best_idx] + reg_lambda)
        
    return best_gain, curr_best_col, curr_best_splitting_val, left_val, right_val

def get_best_sse(X, y, other_predictions, num_cols_other_prediction, features_to_consider, min_samples = 2, delta = 0.1):
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
        B_n_prefix = ((1 + delta * alpha) * np.cumsum(y_sorted) - delta * alpha * np.cumsum(other_predictions_sorted)) / (1 + alpha)
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
            
            left_val = (1 + delta * alpha) * np.mean(left_y) - delta * alpha * np.mean(left_other_predictions)
            right_val = (1 + delta * alpha) * np.mean(right_y) - delta * alpha * np.mean(right_other_predictions)
    
    return best_sse, curr_best_col, curr_best_splitting_val, left_val, right_val

