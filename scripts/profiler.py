import openml
from openml.tasks import list_tasks, TaskType
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from src import GibbsForest
from src import Losses
import logging
import line_profiler as profile
import os

os.environ['LINE_PROFILE'] = "1"

# Load the OpenML dataset
regression_tasks = list_tasks(task_type = TaskType.SUPERVISED_REGRESSION)
small_tasks_ids = []
for task_id, task_value in regression_tasks.items():
    """We want datasets with instances between 5000 and 10000, no missing values, and no symbolic features"""
    if ('NumberOfInstances' in task_value.keys() and task_value['NumberOfInstances'] < 10000 and task_value['NumberOfInstances'] > 5000 and
    'NumberOfMissingValues' in task_value.keys() and task_value['NumberOfMissingValues'] == 0 and 
    'NumberOfSymbolicFeatures' in task_value.keys() and task_value['NumberOfSymbolicFeatures'] == 0):
        small_tasks_ids.append(task_id)
        
logging.getLogger("openml.extensions.sklearn.extension").setLevel(logging.ERROR)
if not hasattr(GibbsForest, '__version__'):
    GibbsForest.__version__ = "0.0.1"  # Use an appropriate version number

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from src import Losses

task_id = small_tasks_ids[0]
task = openml.tasks.get_task(task_id)
X, y = task.get_X_and_y()
print(f"Y mean: {y.mean():.4f}")

gibbs_params = {"eta": 0.1,
            "feature_subsample": 0.9,
            "max_depth": 5,
            "min_samples": 2,
            "n_trees": 100, 
            'row_subsample': 0.9, 
            'warmup_depth': 2, 
            'loss_fn': Losses.LeastSquaresLoss(), 
            'reg_lambda': 0.01,
            'reg_gamma': 0, 
            'tree_eta': 0}

dyna = GibbsForest(**gibbs_params)

xgb_params = {
    'eta': 0.1, 
    'subsample': 0.99, 
    'max_depth': 5, 
    'reg_lambda': 0.1, 
    'gamma': 1
}

xgb = XGBRegressor(**xgb_params)

"""rf_params = {
            "max_depth": 5,
            "max_features": 0.50,
            "min_samples_leaf": 3,
            "n_estimators": 68}
dyna = RandomForestRegressor(**rf_params)"""

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
dyna.fit(X_train, y_train)
train_error = mean_absolute_error(y_train, dyna.predict(X_train))
train_loss = Losses.LeastSquaresLoss()(y_train, dyna.predict(X_train))
test_error = mean_absolute_error(y_test, dyna.predict(X_test))
print(f"Train error: {train_error:.4f}, train loss: {train_loss:.4f}")
print(f"Test error: {test_error:.4f}")
"""booster = dyna.get_booster()
df = booster.trees_to_dataframe()
total_leaves = (df["Feature"] == "Leaf").sum()
print(f"Total leaves: {total_leaves}")"""