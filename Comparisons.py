import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from Dynatree import Dynatree

#Load regression datasets from sklearn
from sklearn.datasets import fetch_california_housing, load_diabetes


def test_other_algorithm(model, X, y, param_dict, folds = 3):
    errors = []
    for random_state in range(2):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = random_state)
        random_search = RandomizedSearchCV(model, param_distributions = param_dict, n_iter = 10, cv = folds, n_jobs = -1, verbose = 2)
        random_search.fit(X_train, y_train)
        y_pred = random_search.predict(X_test)
        errors.append(mean_squared_error(y_test, y_pred))
        
    return errors        


#Load datasets
california_housing = fetch_california_housing() 
X = california_housing.data
y = california_housing.target

models = [Dynatree(), RandomForestRegressor(), XGBRegressor()]
param_dicts = [{'n_trees': [10, 50, 100, 200], 'max_depth': [3, 4, 5], 'window': [1, 2, 5, 10]}, 
               {'n_estimators': [10, 50, 100, 200], 'max_depth': [2, 3, 4, 5, 6, 7, 8, 9, 10], 'min_samples_split': [2, 3, 4, 5, 6, 7, 8, 9, 10]}, 
               {'n_estimators': [10, 50, 100, 200], 'max_depth': [2, 3, 4, 5, 6, 7, 8, 9, 10], 'min_child_weight': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}]
for model, param_dict in zip(models, param_dicts):
    errors = test_other_algorithm(model, X, y, param_dict)
    print("Model: ", model)
    print("Mean error: ", np.mean(errors))

forest = Dynatree(n_trees = 10, window = 4)
forest.fit(X, y)
predictions = forest.predict(X)
squared_error = np.mean((y - predictions)**2)