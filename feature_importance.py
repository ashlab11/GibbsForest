import numpy as np
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.inspection import PartialDependenceDisplay
from scipy.stats import kendalltau
import matplotlib.pyplot as plt
from Dynaforest import Dynatree
from xgboost import XGBRegressor
from sklearn.datasets import make_friedman1
from sklearn.ensemble import RandomForestRegressor
X_train, y_train = make_friedman1(n_samples=200, n_features=10, noise=0, random_state=42)
X_test, y_test = make_friedman1(n_samples=1000, n_features=10, noise=0, random_state=42)

# Define Dynaforest/Random Forest and its hyperparameters
model = Dynatree()

param_distributions = {
    "n_trees": [10, 50, 100],
    "max_depth": [2, 3, 5, 7, 10],
    "feature_subsampling_pct": [0.5, 0.75, 1.0],
    'window': [4, 8, 10]
    #"learning_rate": [0.01, 0.1, 0.3],
    #'reg_alpha': [0, 0.001, 0.005, 0.01, 0.05],
    #'reg_lambda': [0, 0.001, 0.005, 0.01, 0.05],
}

# Use RandomizedSearchCV for hyperparameter optimization
random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_distributions,
    n_iter=4,
    scoring="neg_mean_squared_error",
    cv=3,
    random_state=42,
    n_jobs=-1
)

# Fit the randomized search
random_search.fit(X_train, y_train)

# Best model
best_model = random_search.best_estimator_

"""# Access feature importances
feature_importances = []
num_runs = 5  # Number of cross-validation runs to assess stability

for i in range(num_runs):
    best_model.fit(X_train, y_train)  # Retrain with different seeds if applicable
    feature_importances.append(best_model.feature_importances_)

feature_importances = np.array(feature_importances)

# Calculate Kendall's Tau for feature importance stability
correlations = []
for i in range(num_runs):
    for j in range(i + 1, num_runs):
        tau, _ = kendalltau(feature_importances[i], feature_importances[j])
        correlations.append(tau)

# Average Kendall's Tau
avg_tau = np.mean(correlations)
print(f"Average Kendall's Tau for feature importance stability: {avg_tau:.4f}")
"""

# Partial Dependence Plots (PDPs)
# Identify top features based on importance
top_features = best_model.feature_importances_[[3, 4]]  # Top 3 features

# Generate Partial Dependence Plots
fig, ax = plt.subplots(figsize=(10, 6))
PartialDependenceDisplay.from_estimator(best_model, X_test, features=[0, 1, 2, 3, 4], ax=ax)
plt.tight_layout()
plt.show()

# Output the best hyperparameters
print(f"Best hyperparameters: {random_search.best_params_}")
