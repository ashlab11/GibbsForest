import numpy as np
from line_profiler import profile

@profile
def get_best_sse(X, y, other_predictions, num_cols_other_prediction, features_to_consider, min_samples = 3, epsilon = 0.01):
    alpha = num_cols_other_prediction #naming easier, we'll be using it a lot
    best_sse = np.inf
    curr_best_col = None
    curr_best_splitting_val = None
    #left_val = None
    #right_val = None
    
    for col in features_to_consider: #Going through each column
        sort_idx = np.argsort(X[:, col])
        X_col_sorted = X[:, col][sort_idx]
        y_sorted = y[sort_idx]
        other_predictions_sorted = other_predictions[sort_idx]
        
        
        #Calculating prefix/suffix sums, extremely useful for speeding up calculations
        prefix_sum_y = np.cumsum(y_sorted)
        prefix_sum_y_squared = np.cumsum(np.square(y_sorted))
        prefix_sum_other_predictions = np.cumsum(other_predictions_sorted)
        prefix_sum_other_predictions_squared = np.cumsum(np.square(other_predictions_sorted))
        prefix_sum_multiplication = np.cumsum(np.multiply(y_sorted, other_predictions_sorted))
        prefix_idxs = np.arange(1, len(y_sorted) + 1)
        
        suffix_sum_y = prefix_sum_y[-1] - prefix_sum_y        
        suffix_sum_y_squared = prefix_sum_y_squared[-1] - prefix_sum_y_squared
        suffix_sum_other_predictions = prefix_sum_other_predictions[-1] - prefix_sum_other_predictions
        suffix_sum_other_predictions_squared = prefix_sum_other_predictions_squared[-1] - prefix_sum_other_predictions_squared
        suffix_sum_multiplication = prefix_sum_multiplication[-1] - prefix_sum_multiplication
        suffix_idxs = np.arange(len(y_sorted) - 1, -1, -1)
        
        #We only consider splits with more than min_samples on each side
        valid_mask = np.logical_and(prefix_idxs >= min_samples, suffix_idxs >= min_samples)        
        
        if not np.any(valid_mask):
            #No valid splits
            continue
        
        prefix_sum_y = prefix_sum_y[valid_mask]
        prefix_sum_y_squared = prefix_sum_y_squared[valid_mask]
        prefix_sum_other_predictions = prefix_sum_other_predictions[valid_mask]
        prefix_sum_other_predictions_squared = prefix_sum_other_predictions_squared[valid_mask]
        prefix_sum_multiplication = prefix_sum_multiplication[valid_mask]
        prefix_idxs = prefix_idxs[valid_mask]
        
        suffix_sum_y = suffix_sum_y[valid_mask]
        suffix_sum_y_squared = suffix_sum_y_squared[valid_mask]
        suffix_sum_other_predictions = suffix_sum_other_predictions[valid_mask]
        suffix_sum_other_predictions_squared = suffix_sum_other_predictions_squared[valid_mask]
        suffix_sum_multiplication = suffix_sum_multiplication[valid_mask]
        suffix_idxs = suffix_idxs[valid_mask]
        
        #Look at cumsum.pdf for derivation of these formulas
        left_error = (prefix_sum_y_squared -
                      (2*alpha) / (alpha + 1) * prefix_sum_multiplication -
                      2 / (alpha + 1) * prefix_sum_y**2 / prefix_idxs + 
                        prefix_sum_other_predictions_squared * (alpha / (alpha + 1))**2 + 
                         (2*alpha) / (alpha + 1)**2 * prefix_sum_y * prefix_sum_other_predictions / prefix_idxs + 
                         prefix_sum_y**2 / prefix_idxs * 1/(alpha + 1)**2) 
        # Add the epsilon term later
        # - epsilon * alpha / (prefix_idxs * (alpha + 1)) * prefix_sum_other_predictions
        
        right_error = (suffix_sum_y_squared -
                        (2*alpha) / (alpha + 1) * suffix_sum_multiplication -
                        2 / (alpha + 1) * suffix_sum_y**2 / suffix_idxs +
                        suffix_sum_other_predictions_squared * (alpha / (alpha + 1))**2 +
                        (2*alpha) / (alpha + 1)**2 * suffix_sum_y * suffix_sum_other_predictions / suffix_idxs +
                        suffix_sum_y**2 / suffix_idxs * 1/(alpha + 1)**2) 
        
        sse = left_error + right_error
        
        #We only want to consider the splits when y is at the end of an X value, not in the middle
        #This is because our prefix splits contain X (<= and >), but our suffix splits don't
        #So, we get the last unique indices (or first unique when reversed)
        # Apply the custom unique_with_index
        unique_idx = np.sum(valid_mask) - 1 - np.unique(X_col_sorted[valid_mask][::-1], return_index = True)[1]
        
        best_sse_idx = np.argmin(sse[unique_idx])
        if sse[unique_idx][best_sse_idx] < best_sse:
            best_sse = sse[unique_idx][best_sse_idx]
            curr_best_col = col
            curr_best_splitting_val = X_col_sorted[best_sse_idx] 
            """left_val = prefix_sum_y[best_sse_idx] / prefix_idxs[best_sse_idx] - epsilon * alpha / (
                prefix_idxs[best_sse_idx] * (alpha + 1)) * prefix_sum_other_predictions[best_sse_idx]
            right_val = suffix_sum_y[best_sse_idx] / suffix_idxs[best_sse_idx] - epsilon * alpha / (
                suffix_idxs[best_sse_idx] * (alpha + 1)) * suffix_sum_other_predictions[best_sse_idx]"""
    
    return best_sse, curr_best_col, curr_best_splitting_val

