import numpy as np
from line_profiler import profile
import warnings

@profile
def get_best_sse(X, y, other_predictions, num_cols_other_prediction, features_to_consider, min_samples = 2, delta = 0):
    #print("Function called")
    if len(X) < 2 * min_samples:
        return (np.inf, None, None) #No error reduction possible
    
    alpha = num_cols_other_prediction #naming easier, we'll be using it a lot
    best_sse = np.inf
    curr_best_col = None
    curr_best_splitting_val = None
    
    for col in features_to_consider: #Going through each column
        sort_idx = np.argsort(X[:, col])
        X_col_sorted = X[:, col][sort_idx]
        y_sorted = y[sort_idx]
        other_predictions_sorted = other_predictions[sort_idx]
        
        #We only consider splits with more than min_samples on each side, after accounting for uniqueness
        X_unique_sorted = np.unique(X_col_sorted)
        if len(X_unique_sorted) < 2 * min_samples:
            continue
        #work with prefix ids or smth similar here, should be one line?
        valid_mask = np.logical_and(X_col_sorted >= X_unique_sorted[min_samples - 1], X_col_sorted < X_unique_sorted[-min_samples]) 
        
        if not np.any(valid_mask):
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
        sse = sse[valid_mask]
        
        #We only want to consider the splits when y is at the end of an X value, not in the middle
        #This is because our prefix splits contain X (<= and >), but our suffix splits don't
        #So, we get the last unique indices (or first unique when reversed)
        # Apply the custom unique_with_index
        
        #THIS IS PROBLEM!! INDICES AREN'T WORKING HERE FOR SOME REASON! FIX THIS!
        unique_idx = np.sum(valid_mask) - 1 - np.unique(X_col_sorted[valid_mask][::-1], return_index = True)[1] #Tested, this is correct
                
        try: 
            best_sse_idx_in_sse = np.argmin(sse[unique_idx]) #Get the best sse index in the unique indices
        except:
            print(f"At this point, sse is {sse} and unique_idx is {unique_idx}")
        actual_best_idx = unique_idx[best_sse_idx_in_sse]
        if sse[actual_best_idx] < best_sse:
            best_sse = sse[actual_best_idx]
            curr_best_col = col
            curr_best_splitting_val = X_col_sorted[valid_mask][actual_best_idx]
                   
            if np.sum([X_col_sorted <= curr_best_splitting_val]) < min_samples or np.sum([X_col_sorted > curr_best_splitting_val]) < min_samples:
                print(f"X_sorted is {X_col_sorted}, while valid_mask is {valid_mask}. The length of sse is {len(sse)}, versus {len(X_col_sorted)}")
                print(f"Not enough samples on one side. Samples on left side: {np.sum([X_col_sorted <= curr_best_splitting_val])}, samples on right side: {np.sum([X_col_sorted > curr_best_splitting_val])}")
    
    return best_sse, curr_best_col, curr_best_splitting_val

