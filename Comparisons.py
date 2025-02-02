import numpy as np
import pandas as pd
from scipy.stats.distributions import randint, loguniform
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from Dynaforest import Dynatree

#Load regression datasets from sklearn
from sklearn.datasets import fetch_california_housing, load_diabetes

def test_other_algorithm(model, X, y, param_dict, folds = 5):
    errors = []
    for random_state in range(10):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = random_state)
        random_search = RandomizedSearchCV(model, param_distributions = param_dict, n_iter = 20, cv = folds, n_jobs = -1, verbose = 2)
        random_search.fit(X_train, y_train)
        y_pred = random_search.predict(X_test)
        errors.append(mean_squared_error(y_test, y_pred))
        
    return errors        


#Load datasets
diabetes = load_diabetes() 
X = diabetes.data
y = diabetes.target

models = [Dynatree(), RandomForestRegressor(), XGBRegressor()]
param_dicts = [{'n_trees': randint(10, 200), 'max_depth': randint(3, 10), 'window': [5, 10, 'sqrt', 'log2']}, 
               {'n_estimators': randint(10, 200), 'max_depth': randint(3, 10), 'min_samples_split': randint(2, 10)}, 
               {'n_estimators': randint(10, 200), 'max_depth': randint(3, 10)}]

for model, param_dict in zip(models, param_dicts):
    errors = test_other_algorithm(model, X, y, param_dict)
    print("NEW MODEL HAS FINISHED TRAINING!")
    print("Model: ", model)
    print("Mean error: ", np.mean(errors))
