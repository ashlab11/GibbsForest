import numpy as np
from .Losses import BaseLoss

def check_params(
    X,
    loss_fn, 
    n_trees, 
    max_depth, 
    min_samples, 
    feature_subsample_rf, 
    row_subsample_rf, 
    feature_subsample_g, 
    row_subsample_g, 
    warmup_depth, 
    leaf_eta, 
    tree_eta, 
    reg_lambda, 
    reg_gamma, 
    initial_weight, 
    dropout
):
    """
    Checks parameters, raising ValueErrors along the way
    """
    #--- LOSS FN ---
    if not isinstance(loss_fn, BaseLoss):
        raise TypeError("Loss function must be an extension of BaseLoss")
    
    #--- n_trees ---
    if not isinstance(n_trees, int) or n_trees is None or n_trees < 2:
        raise ValueError('n_trees must be an integer greater than 1')
    
    #--- max_depth ---
    if max_depth is None:
        max_depth = np.inf
    elif not isinstance(max_depth, int):
        raise ValueError('max_depth must be an integer or None')
    
    #--- min_samples ---
    if min_samples is None:
            min_samples = np.inf
    elif not isinstance(min_samples, int):
        raise ValueError('min_samples must be an integer or None')

    #--- subsamples ---
    if feature_subsample_rf == 'sqrt':
        feature_subsample_rf = np.sqrt(len(X[0])) / len(X[0])
        
    if feature_subsample_g == 'sqrt':
        feature_subsample_g = np.sqrt(len(X[0])) / len(X[0])
            
    for subsample in [feature_subsample_rf, feature_subsample_g, row_subsample_rf, row_subsample_g]:
        if not isinstance(subsample, (int, float)):
            raise TypeError("subsample must be a float or convertible to one")
        if subsample > 1 or subsample < 0:
            raise ValueError("subsample must be between 0 and 1")
    

    
    #--- warmup_depth ---
    if warmup_depth == 'half':
        if max_depth == np.inf:
            raise ValueError("warmup_depth cannot be infinite")
        warmup_depth = int(max_depth / 2)
    elif warmup_depth == 'all-but-one':
        if max_depth == np.inf:
            raise ValueError("warmup_depth cannot be infinite")
        warmup_depth = max_depth - 1
    elif not isinstance(warmup_depth, int):
        raise ValueError("warmup_depth must be an integer")

    #--- eta ---
    for eta in [leaf_eta, tree_eta]:
        if not isinstance(eta, (int, float)):
            raise TypeError("eta must be a float or convertible to one")
        if eta > 1 or eta < 0:
            raise ValueError("eta must be between 0 and 1")
        
    #--- reg terms ---
    for reg in [reg_lambda, reg_gamma]:
        if not (isinstance(reg, (int, float))):
            raise TypeError("Regularization terms must be floats or convertible to one")
        if reg < 0:
            raise ValueError("Regularization terms must be positive")
    
    #initial weight
    if initial_weight not in ['parent', 'argmin']:
        raise ValueError("initial weight must be one of ['parent', 'argmin']")
    
    #--- dropout ---
    if not isinstance(dropout, (int, float)):
        raise TypeError("dropout must be a float or convertible to one")
    if dropout > 1 or dropout < 0:
        raise ValueError("dropout must be between 0 and 1")
    
    return (loss_fn, 
    n_trees, 
    max_depth, 
    min_samples, 
    feature_subsample_rf, 
    row_subsample_rf, 
    feature_subsample_g, 
    row_subsample_g, 
    warmup_depth, 
    leaf_eta, 
    tree_eta, 
    reg_lambda, 
    reg_gamma, 
    initial_weight, 
    dropout)