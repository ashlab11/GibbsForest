import os
import json
import logging
import numpy as np
import pandas as pd
from scipy.stats import uniform, randint, loguniform
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import mean_absolute_error, accuracy_score

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
# Import datasets from scikit-learn
from sklearn.datasets import fetch_california_housing, load_breast_cancer, load_iris

# -------------------------
# Import your custom GibbsForest and loss functions
from src import GibbsForest, Losses

# Set logging level (suppress extraneous logs)
logging.getLogger("openml.extensions.sklearn.extension").setLevel(logging.ERROR)

# Create directory for saving best parameter JSON files
if not os.path.exists("param_list"):
    os.makedirs("param_list")

# -------------------------
# Define parameter distributions for tuned models (regression)
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

# -------------------------
# Experiment functions

def run_experiment_regression(X, y, model, param_dist, seed, n_iter=30, tuned=True):
    """
    For tuned models: perform RandomizedSearchCV, return MAE and best params.
    For untuned models (tuned=False): fit with default settings and return MAE.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    
    if tuned:
        search = RandomizedSearchCV(model,
                                    param_distributions=param_dist,
                                    n_iter=n_iter,
                                    cv=5,
                                    random_state=seed,
                                    verbose=3,
                                    n_jobs=-1,
                                    error_score='raise')
        search.fit(X_train, y_train)
        best_params = search.best_params_
        predictions = search.predict(X_test)
    else:
        # Untuned – simply fit the model (for GibbsForest, pass the appropriate loss function)
        model.fit(X_train, y_train)
        best_params = None
        predictions = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, predictions)
    return mae, best_params

def run_experiment_classification(X, y, model, param_dist, seed, n_iter=50, tuned=True):
    """
    For tuned models: perform RandomizedSearchCV, return accuracy and best params.
    For untuned models (tuned=False): fit with default settings and return accuracy.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    
    if tuned:
        search = RandomizedSearchCV(model,
                                    param_distributions=param_dist,
                                    n_iter=n_iter,
                                    cv=5,
                                    random_state=seed,
                                    verbose=3,
                                    n_jobs=-1,
                                    error_score='raise')
        search.fit(X_train, y_train)
        best_params = search.best_params_
        predictions = search.predict(X_test)
    else:
        model.fit(X_train, y_train)
        best_params = None
        predictions = model.predict(X_test)
    
    acc = accuracy_score(y_test, predictions)
    return acc, best_params

# -------------------------
# Define dataset loaders

# Regression datasets
def load_california_housing():
    data = fetch_california_housing(as_frame=True)
    X = data.data
    y = data.target
    return X, y

def load_wine_quality():
    # White wine quality dataset: assumes no missing values.
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
    df = pd.read_csv(url, sep=";")
    X = df.drop("quality", axis=1)
    y = df["quality"]
    return X, y

# Classification datasets
def load_breast_cancer():
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target
    return X, y

def load_iris():
    data = load_iris()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target
    return X, y

# -------------------------
# Set up datasets and models for experiments

regression_datasets = [
    {"name": "California_Housing", "loader": load_california_housing},
    {"name": "Wine_Quality", "loader": load_wine_quality},
]

classification_datasets = [
    {"name": "Breast_Cancer", "loader": load_breast_cancer},
    {"name": "Iris", "loader": load_iris},
]

# For each task, we compare the following:
# GibbsForest (untuned) and tuned versions of:
# RandomForest, XGBoost, LGBM, and CatBoost

# Regression models:
regression_models = [
    {"name": "LGBM", "estimator": LGBMRegressor(verbosity = 0), "param_dist": param_dist_lgbm, "tuned": True},
    {"name": "GibbsForestTuned", "estimator": GibbsForest(loss_fn=Losses.LeastSquaresLoss()), "param_dist": param_dist_gibbs, "tuned": True},
    {"name": "GibbsForestUntuned", "estimator": GibbsForest(loss_fn=Losses.LeastSquaresLoss()), "param_dist": None, "tuned": False},
    {"name": "RandomForest", "estimator": RandomForestRegressor(), "param_dist": param_dist_rf, "tuned": True},
    {"name": "XGBoost", "estimator": XGBRegressor(), "param_dist": param_dist_xgb, "tuned": True},
    {"name": "CatBoost", "estimator": CatBoostRegressor(verbose=0), "param_dist": param_dist_cat, "tuned": True},
]

# Classification models:
classification_models = [
    {"name": "GibbsForest", "estimator": GibbsForest(loss_fn=Losses.CrossEntropyLoss()), "param_dist": param_dist_gibbs, "tuned": True},
    {"name": "GibbsForestUntuned", "estimator": GibbsForest(loss_fn=Losses.CrossEntropyLoss()), "param_dist": None, "tuned": False},
    {"name": "RandomForest", "estimator": RandomForestClassifier(), "param_dist": param_dist_rf, "tuned": True},
    {"name": "XGBoost", "estimator": XGBClassifier(use_label_encoder=False, eval_metric='logloss'), "param_dist": param_dist_xgb, "tuned": True},
    {"name": "LGBM", "estimator": LGBMClassifier(), "param_dist": param_dist_lgbm, "tuned": True},
    {"name": "CatBoost", "estimator": CatBoostClassifier(verbose=0), "param_dist": param_dist_cat, "tuned": True},
]

# -------------------------
# Run experiments
results = {"regression": {}, "classification": {}}

# Regression experiments
for ds in regression_datasets:
    ds_name = ds["name"]
    print(f"\n=== Regression dataset: {ds_name} ===")
    X, y = ds["loader"]()
    results["regression"][ds_name] = {}
    
    for model_info in regression_models:
        model_name = model_info["name"]
        print(f"\n-- Model: {model_name} --")
        seed_scores = []
        seed_params = {}
        
        for s in range(15):
            # Use a clone of the estimator to avoid interference across seeds.
            from sklearn.base import clone
            est = clone(model_info["estimator"])
            mae, best_params = run_experiment_regression(X, y, est,
                                                         param_dist=model_info["param_dist"],
                                                         seed=s,
                                                         n_iter=30,
                                                         tuned=model_info["tuned"])
            print(f"Seed {s}: MAE = {mae:.4f}")
            seed_scores.append(mae)
            if model_info["tuned"]:
                seed_params[str(s)] = best_params
                # Save best parameters to JSON
                fname = f"param_list/{ds_name}_{model_name}_seed{s}.json"
                with open(fname, "w") as f:
                    json.dump(best_params, f, indent=4)
        avg_score = np.mean(seed_scores)
        std_score = np.std(seed_scores)
        results["regression"][ds_name][model_name] = {"MAE_mean": avg_score,
                                                      "MAE_std": std_score,
                                                      "best_params": seed_params if model_info["tuned"] else None}
        print(f"{model_name} on {ds_name}: Average MAE = {avg_score:.4f} ± {std_score:.4f}")

# Classification experiments
for ds in classification_datasets:
    ds_name = ds["name"]
    print(f"\n=== Classification dataset: {ds_name} ===")
    X, y = ds["loader"]()
    results["classification"][ds_name] = {}
    
    for model_info in classification_models:
        model_name = model_info["name"]
        print(f"\n-- Model: {model_name} --")
        seed_scores = []
        seed_params = {}
        
        for s in range(15):
            from sklearn.base import clone
            est = clone(model_info["estimator"])
            acc, best_params = run_experiment_classification(X, y, est,
                                                             param_dist=model_info["param_dist"],
                                                             seed=s,
                                                             n_iter=30,
                                                             tuned=model_info["tuned"])
            print(f"Seed {s}: Accuracy = {acc:.4f}")
            seed_scores.append(acc)
            if model_info["tuned"]:
                seed_params[str(s)] = best_params
                fname = f"param_list/{ds_name}_{model_name}_seed{s}.json"
                with open(fname, "w") as f:
                    json.dump(best_params, f, indent=4)
        avg_score = np.mean(seed_scores)
        std_score = np.std(seed_scores)
        results["classification"][ds_name][model_name] = {"Accuracy_mean": avg_score,
                                                          "Accuracy_std": std_score,
                                                          "best_params": seed_params if model_info["tuned"] else None}
        print(f"{model_name} on {ds_name}: Average Accuracy = {avg_score:.4f} ± {std_score:.4f}")

# -------------------------
# Save overall results to a JSON file
with open("param_list/overall_results.json", "w") as f:
    json.dump(results, f, indent=4)

# -------------------------
# Print LaTeX table skeletons for inclusion in your paper.
# These tables have placeholders ("--") for you to later fill in with your results.

latex_regression = r"""
\begin{table}[ht]
\centering
\caption{Regression Results (MAE)}
\begin{tabular}{lccccc}
\hline
Dataset & GF (untuned) & RF (tuned) & XGB (tuned) & LGBM (tuned) & CatBoost (tuned) \\
\hline
California Housing & -- & -- & -- & -- & -- \\
Wine Quality       & -- & -- & -- & -- & -- \\
\hline
\end{tabular}
\label{tab:regression_results}
\end{table}
"""

latex_classification = r"""
\begin{table}[ht]
\centering
\caption{Classification Results (Accuracy)}
\begin{tabular}{lccccc}
\hline
Dataset & GF (untuned) & RF (tuned) & XGB (tuned) & LGBM (tuned) & CatBoost (tuned) \\
\hline
Breast Cancer & -- & -- & -- & -- & -- \\
Iris          & -- & -- & -- & -- & -- \\
\hline
\end{tabular}
\label{tab:classification_results}
\end{table}
"""

print("\nLaTeX Table for Regression Results:")
print(latex_regression)
print("\nLaTeX Table for Classification Results:")
print(latex_classification)

print("\nExperiment completed. Overall results saved in 'param_list/overall_results.json'.")
