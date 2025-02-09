import openml
import pandas as pd
import numpy as np
from scipy.stats import uniform, randint, loguniform
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import mean_absolute_error
from openml.tasks import list_tasks, TaskType
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from src import GibbsForest
import logging
import json

seed = 42

# Load the OpenML task list
regression_tasks = list_tasks(task_type = TaskType.SUPERVISED_REGRESSION)
small_tasks_ids = []
for task_id, task_value in regression_tasks.items():
    """We want datasets with instances between 5000 and 10000, no missing values, and no symbolic features"""
    if ('NumberOfInstances' in task_value.keys() and task_value['NumberOfInstances'] < 10000 and task_value['NumberOfInstances'] > 5000 and
    'NumberOfMissingValues' in task_value.keys() and task_value['NumberOfMissingValues'] == 0 and 
    'NumberOfSymbolicFeatures' in task_value.keys() and task_value['NumberOfSymbolicFeatures'] == 0):
        small_tasks_ids.append(task_id)
        
#Fix warnings
logging.getLogger("openml.extensions.sklearn.extension").setLevel(logging.ERROR)
if not hasattr(GibbsForest, '__version__'):
    GibbsForest.__version__ = "0.0.1"  # Use an appropriate version number

#Create parameter distributions to pull from
param_dist_rf = {
    'n_estimators': randint(20, 100),
    'max_depth': randint(2, 6),
    'min_samples_split': randint(2, 10),
    'min_samples_leaf': randint(1, 50),
    'max_features': uniform(0, 1),
}

param_dist_xgb = {
    'n_estimators': randint(20, 100),
    'max_depth': randint(2, 6),
    'learning_rate': loguniform(0.001, 1),
    'subsample': uniform(0, 1)
}

param_dist_dynatree = {
    'n_trees': randint(20, 100),
    'max_depth': randint(2, 6),
    'min_samples': randint(2, 10),
    'delta': loguniform(0.001, 1),
    'window': ['sqrt', 'log2', 4, 8, 16],
}

def run_experiment(X, y, model, param_dist, seed, n_iter = 20):
    """Choose best hyperparameters for a model on a task"""
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = seed)
    rg = RandomizedSearchCV(model, param_distributions = param_dist, n_iter = n_iter, cv = 3, random_state = seed, verbose = 2)
    rg.fit(X_train, y_train)
    best_params = rg.best_params_
    test_error = mean_absolute_error(y_test, rg.predict(X_test))
    
    return test_error, best_params
    
    
    
num_datasets = 1
error_array = np.zeros((num_datasets, 3))
param_array = [[{} for _ in range(3)] for _ in range(num_datasets)]
for run_num in range(num_datasets):
    task_id = small_tasks_ids[run_num]
    task = openml.tasks.get_task(task_id)
    X, y = task.get_X_and_y()
    
    for idx, (model, param_grid) in enumerate(zip([RandomForestRegressor(), XGBRegressor(), Dynatree()], [param_dist_rf, param_dist_xgb, param_dist_dynatree])):
        maes, params = run_experiment(X, y, model, param_grid, seed = seed + run_num)
        error_array[run_num, idx] = maes
        param_array[run_num][idx] = params
        
        
#Save error and param arrays
np.save('errors.npy', error_array)

# Write the data to a JSON file
with open("params.json", "w") as f:
    json.dump(param_array, f, indent=4)
    
    
    
    