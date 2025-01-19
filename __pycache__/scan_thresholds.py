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
