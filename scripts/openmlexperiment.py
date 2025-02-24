import openml
from openml.tasks import list_tasks, TaskType
import os
import json
import logging
import numpy as np
import pandas as pd
from scipy.stats import uniform, randint, loguniform, ttest_rel
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score

# -------------------------
# Import models for regression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# Import models for classification
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# -------------------------
# Importing GibbsForest
from src import GibbsForest, Losses

# --- PARAM DISTRIBUTIONS ---#
param_dist_rf = {
    'n_estimators': randint(20, 150),
    'max_depth': randint(3, 6),
    'min_samples_split': randint(2, 10),
    'min_samples_leaf': randint(1, 50),
    'max_features': uniform(0.1, 0.9),
}

param_dist_xgb = {
    'n_estimators': randint(20, 150),
    'max_depth': randint(3, 6),
    'learning_rate': loguniform(0.001, 0.5),
    'subsample': uniform(0.5, 0.5),
    'colsample_bytree': uniform(0.5, 0.5),
    'reg_alpha': loguniform(0.001, 1),
    'reg_lambda': loguniform(0.001, 1),
}

param_dist_lgbm = {
    'n_estimators': randint(20, 150),
    'max_depth': [-1],
    'learning_rate': loguniform(0.001, 0.5),
    'num_leaves': randint(8, 64),
    'subsample': uniform(0.5, 0.5),
    'colsample_bytree': uniform(0.5, 0.5)
}

param_dist_cat = {
    'iterations': randint(20, 150),
    'depth': randint(3, 6),
    'learning_rate': loguniform(0.001, 0.5),
    'l2_leaf_reg': loguniform(0.001, 10), 
    'logging_level': ['Silent']
}

param_dist_gibbs = {
    "n_trees": randint(20, 150),
    'max_depth': randint(3, 6),
    'leaf_eta': uniform(0.05, 0.15), 
    'tree_eta': uniform(0, 0.1),
    'feature_subsample_rf': uniform(0.5, 0.5), 
    'row_subsample_rf': uniform(0.5, 0.5), 
    'warmup_depth': [1, 2, 'all-but-one']
}

#--- MODELS ---#
models = {
    'lgbm': (LGBMRegressor(verbosity = -1, n_jobs = 1), param_dist_lgbm),
    'cat': (CatBoostRegressor(logging_level='Silent'), param_dist_cat),
    'rf': (RandomForestRegressor(), param_dist_rf),
    'xgb': (XGBRegressor(), param_dist_xgb),
    'gibbs': (GibbsForest(), param_dist_gibbs)
}

benchmark_suite = openml.study.get_suite(297)

#---- RUNNING EXPERIMENT ----#
num_seeds = 20

#Create a dictionary to store the results
results = {}

def get_max_depth(tree, current_depth=0):
    if current_depth == 0:
        tree = tree['tree_structure']
    # If the node has no children, it's a leaf: return the current depth.
    if "left_child" not in tree and "right_child" not in tree:
        return current_depth
    # Recurse on left and right children if they exist.
    left_depth = get_max_depth(tree["left_child"], current_depth + 1) if "left_child" in tree else current_depth
    right_depth = get_max_depth(tree["right_child"], current_depth + 1) if "right_child" in tree else current_depth
    return max(left_depth, right_depth)

for task_id in benchmark_suite.tasks[1:]:
    task = openml.tasks.get_task(task_id)
    dataset = task.get_dataset()
    name = dataset.name
    obs = dataset.qualities['NumberOfInstances']
    features = dataset.qualities['NumberOfFeatures'] 
    print(f"===== DATASET {name} with {obs} observations and {features} features ====")
    
    #Get X and y
    X, y = task.get_X_and_y(dataset_format='dataframe')
    X = X.to_numpy()
    y = y.to_numpy()
    errors = np.zeros((len(models), num_seeds))
    
    for i, (model_name, (model, param_dist)) in enumerate(models.items()):
        print(f"----- MODEL: {model_name} -----")
        for seed in range(num_seeds):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
            cv = RandomizedSearchCV(model, param_dist, n_iter=30, cv=5, n_jobs=-1, random_state=seed, scoring='neg_mean_squared_error')
            cv.fit(X_train, y_train)
            if model_name == 'lgbm':
                lgb_model = cv.best_estimator_
                model_dump = lgb_model.booster_.dump_model()
                trees = model_dump['tree_info']
                print(f"Max depth of LGBM trees: {[get_max_depth(tree) for tree in trees]}")
                
            y_pred = cv.predict(X_test)
            error = mean_squared_error(y_test, y_pred)
            errors[i, seed] = error
            print(f"Seed {seed} - Error: {error}")
            
    #Store the results
    results[task_id] = errors
    
    #Get means, calculate t-test and print results
    means = errors.mean(axis=1)
    argsort_idxs = np.argsort(means)
    best_model = errors[argsort_idxs[0]]
    second_best_model = errors[argsort_idxs[1]]
    _, p_value = ttest_rel(best_model, second_best_model)
    print(f"Best model: {list(models.keys())[argsort_idxs[0]]} with mean error {best_model.mean()} +/- {best_model.std()}")
    print(f"P-value of best model over second best model: {p_value}")


# -------------------------
# Save overall results to a JSON file
with open("param_list/overall_results.json", "w") as f:
    json.dump(results, f, indent=4)
