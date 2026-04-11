from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import numpy as np


def get_models_and_params():
    """
    Returns models and their hyperparameter grids.
    """

    models = {}

    # Logistic Regression
    models["LogisticRegression"] = {
        "model": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000))
        ]),
        "params": {
            "model__C": [0.01, 0.1, 1, 10]
        }
    }

    # Random Forest
    models["RandomForest"] = {
        "model": RandomForestClassifier(random_state=42),
        "params": {
            "n_estimators": [100, 200],
            "max_depth": [None, 5, 10]
        }
    }

    # XGBoost
    models["XGBoost"] = {
        "model": xgb.XGBClassifier(
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        ),
        "params": {
            "n_estimators": [100, 200],
            "learning_rate": [0.01, 0.1],
            "max_depth": [3, 5]
        }
    }

    return models


def train_and_select_best(X_train, y_train):
    """
    Trains models using GridSearchCV and selects best model.
    """

    models = get_models_and_params()

    best_model = None
    best_score = 0
    best_name = ""

    print("\n===== MODEL COMPARISON WITH HYPERPARAMETER TUNING =====")

    for name, model_info in models.items():

        grid_search = GridSearchCV(
            model_info["model"],
            model_info["params"],
            cv=5,
            scoring="accuracy",
            n_jobs=-1
        )

        grid_search.fit(X_train, y_train)

        mean_score = grid_search.best_score_

        print(f"{name}: {mean_score:.4f}")
        print(f"Best Params: {grid_search.best_params_}\n")

        if mean_score > best_score:
            best_score = mean_score
            best_model = grid_search.best_estimator_
            best_name = name

    print(f"\nBest Model Selected: {best_name} (Score: {best_score:.4f})")

    return best_model, best_name, best_score
