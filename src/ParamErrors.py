import numpy as np
from .Losses import BaseLoss
from typing import SupportsFloat

def check_params(
    loss_fn,
    n_trees,
    max_depth, 
    min_samples,
    feature_subsample, 
    row_subsample, 
    warmup_depth, 
    eta, 
    reg_lambda, 
    reg_gamma, 
    initial_weight
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

    #--- feature_subsample ---
    if not isinstance(feature_subsample, SupportsFloat):
        raise TypeError("feature_subsample must be a float or convertible to one")
    if feature_subsample > 1 or feature_subsample < 0:
        raise ValueError("feature_subsample must be between 0 and 1")
    
    #--- row_subsample ---
    if not isinstance(row_subsample, SupportsFloat):
        raise TypeError("row_subsample must be a float or convertible to one")
    if row_subsample > 1 or row_subsample < 0:
        raise ValueError("row_subsample must be between 0 and 1")

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
    if not isinstance(eta, SupportsFloat):
        raise TypeError("eta must be a float or convertibel to one")
    if eta > 1 or eta < 0:
        raise ValueError("eta must be between 0 and 1")
    
    #--- reg terms ---
    if not (isinstance(reg_gamma, SupportsFloat) and isinstance(reg_lambda, SupportsFloat)):
        raise ValueError('Regularization terms must be floats or convertible to one')
    elif reg_gamma < 0 or reg_lambda < 0:
        raise ValueError('Regularization Terms must be positive')
    
    #initial weight
    if initial_weight not in ['parent', 'argmin']:
        raise ValueError("initial weight must be one of ['parent', 'argmin']")
    
    return loss_fn, n_trees, max_depth, min_samples, feature_subsample, row_subsample, warmup_depth, eta, reg_lambda, reg_gamma, initial_weight