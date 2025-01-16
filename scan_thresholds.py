from numba import njit
import numpy as np

#Using numba speeds up by factor of 3x
def get_best_sse3(X, y, other_predictions, num_cols_other_predictions, min_samples = 3):
    best_sse = np.inf
    curr_best_col = None
    curr_best_splitting_val = None
    for col in range(len(X[0])): #Going through each column
            unique_vals_sorted = np.unique(X[:, col])
            for val in unique_vals_sorted: #Going through each unique value, and getting predictions based on means of split
                below_indices = X[:, col] <= val
                above_indices = X[:, col] > val
                
                if np.sum(below_indices) < min_samples or np.sum(above_indices) < min_samples:
                    continue
                
                #GD here
                below_mean, above_mean = np.mean(y[below_indices]), np.mean(y[above_indices])
                
                #Combining predictions from other trees
                below_predictions = (other_predictions[below_indices] * num_cols_other_predictions + below_mean) / (num_cols_other_predictions + 1)
                above_predictions = (other_predictions[above_indices] * num_cols_other_predictions + above_mean) / (num_cols_other_predictions + 1)
                below_error = np.sum((y[X[:, col] <= val] - below_predictions)**2)
                above_error = np.sum((y[X[:, col] > val] - above_predictions)**2)
                
                sse = below_error + above_error
                #Update best error reduction
                if sse < best_sse:
                    best_sse = sse
                    curr_best_col = col
                    curr_best_splitting_val = val
        
    return best_sse, curr_best_col, curr_best_splitting_val

def get_best_sse(X, y, other_predictions, num_cols_other_prediction, min_samples = 3):
    alpha = num_cols_other_prediction #naming easier, we'll be using it a lot
    best_sse = np.inf
    curr_best_col = None
    curr_best_splitting_val = None
    
    for col in range(len(X[0])): #Going through each column
        sort_idx = np.argsort(X[:, col])
        X_col_sorted = X[:, col][sort_idx]
        y_sorted = y[sort_idx]
        other_predictions_sorted = other_predictions[sort_idx]
        
        #Calculating prefix/suffix sums, extremely useful for speeding up calculations
        prefix_sum_y = np.cumsum(y_sorted)
        prefix_sum_y_squared = np.cumsum(y_sorted**2)
        prefix_sum_other_predictions = np.cumsum(other_predictions_sorted)
        prefix_sum_other_predictions_squared = np.cumsum(other_predictions_sorted**2)
        prefix_sum_multiplication = np.cumsum(y_sorted * other_predictions_sorted)
        prefix_idxs = np.arange(1, len(y_sorted) + 1)
        
        suffix_sum_y = np.sum(y_sorted) - prefix_sum_y
        suffix_sum_y_squared = np.sum(y_sorted**2) - prefix_sum_y_squared
        suffix_sum_other_predictions = np.sum(other_predictions_sorted) - prefix_sum_other_predictions
        suffix_sum_other_predictions_squared = np.sum(other_predictions_sorted**2) - prefix_sum_other_predictions_squared
        suffix_sum_multiplication = np.sum(y_sorted * other_predictions_sorted) - prefix_sum_multiplication
        suffix_idxs = np.arange(len(y_sorted) - 1, -1, -1)
        
        #We only consider splits with more than min_samples on each side
        valid_mask = np.logical_and(prefix_idxs >= min_samples, suffix_idxs >= min_samples)
        
        if np.sum(valid_mask) == 0:
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
        unique_idx = np.sum(valid_mask) - 1 - np.unique(X_col_sorted[valid_mask][::-1], return_index = True)[1]
        
        best_sse_idx = np.argmin(sse[unique_idx])
        if sse[unique_idx][best_sse_idx] < best_sse:
            best_sse = sse[unique_idx][best_sse_idx]
            curr_best_col = col
            curr_best_splitting_val = X_col_sorted[best_sse_idx] 
    
    return best_sse, curr_best_col, curr_best_splitting_val

@njit
def get_best_sse2(X, y, other_predictions, num_cols_other_prediction):
    """Fix this algorithm so it takes in the other predictions and the number of columns in the other predictions"""
    best_sse = np.inf
    curr_best_col = None
    curr_best_splitting_val = None
    #precomputing prefix sums, to speed up calculation
    for col in range(len(X[0])):
        sort_idx = np.argsort(X[:, col])
        X_col_sorted = X[:, col][sort_idx]
        y_sorted = y[sort_idx]
        
        prefix_sums = np.cumsum(y_sorted)
        prefix_sums_squared = np.cumsum(y_sorted**2)
        suffix_sums = np.sum(y_sorted) - prefix_sums
        suffix_sums_squared = np.sum(y_sorted**2) - prefix_sums_squared
        
        #We want left split to be all values with x <= val, and right split to be all values with x > val
        for i in range(len(y_sorted) - 1):
            if X_col_sorted[i] == X_col_sorted[i + 1]:
                #We don't want to split on the same value, wait until we get to the final one
                continue
            
            sum_left = prefix_sums[i]
            sumSq_left = prefix_sums_squared[i]
            count_left = i + 1
            sse_left = sumSq_left - (sum_left**2 / count_left)
            
            sum_right = suffix_sums[i]
            sumSq_right = suffix_sums_squared[i]
            count_right = len(y_sorted) - i - 1
            sse_right = sumSq_right - (sum_right**2 / count_right)
            
            sse = sse_left + sse_right
            if sse < best_sse:
                best_sse = sse
                val = X_col_sorted[i]
                curr_best_col = col
                curr_best_splitting_val = val
                
        return best_sse, curr_best_col, curr_best_splitting_val