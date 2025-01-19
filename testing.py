import numpy as np
import time
from collections import deque
from Dynaforest import Dynatree
from sklearn.utils.estimator_checks import check_estimator
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from line_profiler import profile



def create_regression_data(n_points, n_features):
    X = np.random.rand(n_points, n_features)
    y = X[:, 0] * 5 + np.sign(X[:, 1]) + np.sin(X[:, 2]) + 1 + np.random.randn(n_points)
    return X, y


X_train, y_train = create_regression_data(1000, 3)
X_test, y_test = create_regression_data(100, 3)

X, y = make_regression(
            n_samples=20000,
            n_features=10,
            n_informative=3,
            bias=5.0,
            noise=10, 
            random_state=42
        )
X = StandardScaler().fit_transform(X)

start = time.time()
forest = Dynatree(n_trees = 100, window = 15, max_depth = 4, feature_subsampling_pct=0.2)

def main():
    forest.fit(X, y)
    score = forest.score(X, y)
    
    predictions = [tree.predict(X) for tree in forest._trees]
    ensemble_predictions = np.mean(predictions, axis = 0)
    average_covariance = np.mean(np.corrcoef(predictions))
    
    squared_bias = np.mean((np.mean(predictions, axis = 0) - y)**2)
    variances = np.mean((predictions - ensemble_predictions)**2, axis = 0)
    variance = np.mean(variances)
    
    pairwise_corrs = []
    for i in range(len(predictions)):
        for j in range(i + 1, len(predictions)):
            pairwise_corrs.append(np.corrcoef(predictions[i], predictions[j])[0, 1])
            
    average_covariance = np.mean(pairwise_corrs)
    total_error = squared_bias + 1 / len(forest._trees) * variance + (len(forest._trees) - 1) / len(forest._trees) * average_covariance * variance

    print(f"---- DYNAFOREST ---")
    print(f"score, {score}")
    #Bias
    print(f"Squared Bias: {squared_bias}")
    print(f"Variance: {variance}")
    print(f"Covariance: {average_covariance}")
    print(f"Total Error: {total_error}")
    
    return forest

def rforest():
    forest = RandomForestRegressor(n_estimators = 100, max_depth = 4, max_features=0.2)
    forest.fit(X, y)
    score = forest.score(X, y)
    
    predictions = [tree.predict(X) for tree in forest.estimators_]
    ensemble_predictions = np.mean(predictions, axis = 0)
    average_covariance = np.mean(np.corrcoef(predictions))
    
    squared_bias = np.mean((np.mean(predictions, axis = 0) - y)**2)
    variances = np.mean((predictions - ensemble_predictions)**2, axis = 0)
    variance = np.mean(variances)
    
    pairwise_corrs = []
    for i in range(len(predictions)):
        for j in range(i + 1, len(predictions)):
            pairwise_corrs.append(np.corrcoef(predictions[i], predictions[j])[0, 1])
            
    average_covariance = np.mean(pairwise_corrs)
    total_error = squared_bias + 1 / len(forest.estimators_) * variance + (len(forest.estimators_) - 1) / len(forest.estimators_) * average_covariance * variance

    print(f"---- RANDOM FOREST ---")
    print(f"score, {score}")
    #Bias
    print(f"Squared Bias: {squared_bias}")
    print(f"Variance: {variance}")
    print(f"Covariance: {average_covariance}")
    print(f"Total Error: {total_error}")
    return forest

dynaforest = main()
rforest = rforest()
#Calculate biases and average covariance between trees


#print(f"time taken was {time.time() - start}")