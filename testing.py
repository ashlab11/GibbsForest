import numpy as np
import time
from collections import deque
from Dynatree2 import Dynatree
from sklearn.utils.estimator_checks import check_estimator
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
import xgboost as xgb


def create_regression_data(n_points, n_features):
    X = np.random.rand(n_points, n_features)
    y = X[:, 0] * 5 + np.sign(X[:, 1]) + np.sin(X[:, 2]) + 1 + np.random.randn(n_points)
    return X, y


X_train, y_train = create_regression_data(1000, 3)
X_test, y_test = create_regression_data(100, 3)


X, y = make_regression(
            n_samples=200,
            n_features=10,
            n_informative=1,
            bias=5.0,
            noise=10, 
            random_state=42
        )
X = StandardScaler().fit_transform(X)

start = time.time()
forest = Dynatree(n_trees = 10, window = 4, max_depth = 3)
forest.fit(X, y)
#forest = xgb.XGBRegressor(n_estimators = 100, max_depth = 3, min_child_weight = 1)
#forest.fit(X, y)
predictions = forest.predict(X)
print(forest.trees)

print(f"time taken was {time.time() - start}")