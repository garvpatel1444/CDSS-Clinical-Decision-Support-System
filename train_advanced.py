import mlflow
import mlflow.sklearn

import joblib
from ml_pipeline.data_loader import load_dataset
from ml_pipeline.model_trainer import train_and_select_best
from ml_pipeline.evaluator import evaluate_model

# ============================
# SELECT DISEASE HERE
# ============================

DATASET_PATH = "data/heart.csv"
TARGET_COLUMN = "target"
MODEL_SAVE_PATH = "models/heart_best_model.pkl"

# For Diabetes:
# DATASET_PATH = "data/diabetes.csv"
# TARGET_COLUMN = "Outcome"
# MODEL_SAVE_PATH = "models/diabetes_best_model.pkl"

# ============================
# LOAD DATA
# ============================

X_train, X_test, y_train, y_test = load_dataset(
    DATASET_PATH,
    TARGET_COLUMN
)

# ============================
# TRAIN & SELECT BEST MODEL
# ============================

best_model, best_name, best_score = train_and_select_best(
    X_train,
    y_train
)
# ============================
# MLFLOW LOGGING
# ============================

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Multi-Disease-CDSS")

with mlflow.start_run():

    mlflow.log_param("dataset", DATASET_PATH)
    mlflow.log_param("model_type", best_name)
    mlflow.log_metric("cv_score", best_score)

    # Log test accuracy
    from sklearn.metrics import accuracy_score
    y_pred = best_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    mlflow.log_metric("test_accuracy", test_accuracy)

    # Log model
    mlflow.sklearn.log_model(
    sk_model=best_model,
    artifact_path="model",
    registered_model_name="CDSS_Model"
)


    print("\nMLflow logging completed.")

# ============================
# EVALUATE
# ============================

evaluate_model(best_model, X_test, y_test)

# ============================
# SAVE MODEL
# ============================

joblib.dump(best_model, MODEL_SAVE_PATH)

print(f"\nModel saved successfully as {MODEL_SAVE_PATH}")

# ============================
# SHAP EXPLAINABILITY
# ============================

from ml_pipeline.explainer import generate_shap_explanation
import pandas as pd
import numpy as np

print("\n===== SHAP EXPLANATION FOR ONE SAMPLE =====")

sample = X_test.iloc[[0]]

try:
    shap_values = generate_shap_explanation(best_model, sample)

    if isinstance(shap_values, list):
        values = shap_values[1][0]
    else:
        values = shap_values[0]

    feature_importance = dict(zip(sample.columns, values))

    sorted_features = sorted(
        feature_importance.items(),
        key=lambda x: abs(x[1]),
        reverse=True
    )

    print("\nTop 5 Important Features:")
    for feature, importance in sorted_features[:5]:
        print(f"{feature}: {importance:.4f}")

except Exception as e:
    print("SHAP failed:", e)

