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
    'subsample': uniform(0, 1),
    'colsample_bytree': uniform(0, 1),
    'reg_alpha': loguniform(0.001, 1), 
    'reg_lambda': loguniform(0.001, 1), 
    'gamma': loguniform(0.001, 1)
}

param_dist_gibbs = {
    'n_trees': randint(20, 100),
    'max_depth': randint(3, 6),
    'min_samples': randint(2, 10),
    'feature_subsample': uniform(0, 1),
    'row_subsample': uniform(0, 1),
    'reg_lambda': loguniform(0.001, 1), 
    'reg_gamma': loguniform(0.001, 1), 
    'warmup_depth': [1, 2, 3],
    'eta': uniform(0.05, 0.2), 
    'initial_weight': ['parent', 'argmin']
}

def run_experiment(X, y, model, param_dist, seed, n_iter = 30):
    """Choose best hyperparameters for a model on a task"""
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = seed)
    rg = RandomizedSearchCV(model, param_distributions = param_dist, n_iter = n_iter, cv = 5, random_state = seed, verbose = 1, 
                            n_jobs = -1, error_score = 'raise')
    rg.fit(X_train, y_train)
    best_params = rg.cv_results_
    test_error = mean_absolute_error(y_test, rg.predict(X_test))
    
    return test_error, best_params
    
    
num_datasets = 10
error_array = np.zeros((num_datasets, 3))
for run_num in range(num_datasets):
    task_id = small_tasks_ids[run_num]
    task = openml.tasks.get_task(task_id)
    X, y = task.get_X_and_y()
    
    for idx, (model, param_grid) in enumerate(zip([RandomForestRegressor(), XGBRegressor(), GibbsForest()], [param_dist_rf, param_dist_xgb, param_dist_gibbs])):
        maes, params = run_experiment(X, y, model, param_grid, seed = seed + run_num)
        error_array[run_num, idx] = maes
        
        if idx == 1:
            params_df = pd.DataFrame.from_dict(params)
            params_df.to_csv(f"param_list/Dataset_{run_num}")
        
        
#Save error array
np.save('errors.npy', error_array)
    
print(f"Errors for Random forest: {np.mean(error_array[:, 0])}")
print(f"Errors for XGBoost: {np.mean(error_array[:, 1])}")
print(f"Errors for Dynatree: {np.mean(error_array[:, 2])}")

print(f"Percent of time Dynatree is better than Random Forest: {np.mean(error_array[:, 2] < error_array[:, 0])}")
print(f"Percent of time Dynatree is better than XGBoost: {np.mean(error_array[:, 2] < error_array[:, 1])}")

print(f"Average relative improvement of Dynatree over Random Forest: {np.mean((error_array[:, 0] - error_array[:, 2]) / error_array[:, 0])}")
print(f"Average relative improvement of Dynatree over XGBoost: {np.mean((error_array[:, 1] - error_array[:, 2]) / error_array[:, 1])}")    