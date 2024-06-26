import optuna
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

data = fetch_openml(name="adult")
X = pd.get_dummies(data["data"])
y = [1 if d == ">50K" else 0 for d in data["target"]]

"""
clf = RandomForestClassifier(
    max_depth = 8,
    min_samples_split = 0.5,
)
"""

def objective(trial):
    clf = RandomForestClassifier(
        max_depth = trial.suggest_int("max_depth",2,32,),
        min_samples_split = trial.suggest_float("min_samples_split",0,1),
    )
    score = cross_val_score(clf, X, y, cv=3)
    accuracy = score.mean()
    return accuracy

"""
score = cross_val_score(clf, X, y, cv=3)
accuracy = score.mean()
print(f"Accuracy: {accuracy}")
"""

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials = 100)
print(f"Best objective value: {study.best_value}")
print(f"Best parameter: {study.best_params}")
