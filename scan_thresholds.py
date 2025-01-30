import numpy as np
from line_profiler import profile

@profile
def get_best_sse(X, y, other_predictions, num_cols_other_prediction, features_to_consider, min_samples = 3):
    #print("Function called")
    if len(X) < 2 * min_samples:
        return (np.inf, -1, -1) #No error reduction possible
    
    alpha = num_cols_other_prediction #naming easier, we'll be using it a lot
    best_sse = np.inf
    curr_best_col = None
    curr_best_splitting_val = None
    
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
        suffix_idxs = prefix_idxs[-1] - prefix_idxs
        
        #We only consider splits with more than min_samples on each side, after accounting for uniqueness
        #FIX THIS PART! you need to account for uniqueness in the indices, otherwise you'll end up with not enough values on each side
        X_unique_sorted = np.unique(X_col_sorted)
        if len(X_unique_sorted) < 2 * min_samples:
            continue
        #work with prefix ids or smth similar here, should be one line?
        valid_mask = np.logical_and(X_col_sorted >= X_unique_sorted[min_samples - 1], X_col_sorted < X_unique_sorted[-min_samples]) 
        
        if not np.any(valid_mask):
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

