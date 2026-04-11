import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

# -----------------------------
# Load dataset
# -----------------------------

df = pd.read_csv("diabetes.csv")

# Outcome: 1 = diabetic, 0 = non-diabetic
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# -----------------------------
# Split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -----------------------------
# Model
# Improved hyperparameters for better accuracy and probability calibration:
#   - More estimators (500 vs 300) → smoother probability estimates
#   - Slightly deeper trees (max_depth=10) → captures more patterns
#   - min_samples_leaf=2 → reduces overfitting on noisy leaves
#   - class_weight='balanced' → handles class imbalance (500 non-diabetic vs 268 diabetic)
# -----------------------------

model = RandomForestClassifier(
    n_estimators=500,
    max_depth=10,
    min_samples_split=3,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42
)

# -----------------------------
# Train
# -----------------------------

model.fit(X_train, y_train)

# -----------------------------
# Evaluate
# -----------------------------

pred = model.predict(X_test)
proba = model.predict_proba(X_test)[:, 1]  # probability of class 1 (diabetic)

accuracy = accuracy_score(y_test, pred)
roc_auc = roc_auc_score(y_test, proba)

print("Diabetes Accuracy :", accuracy)
print("Diabetes ROC-AUC  :", roc_auc)
print()
print(classification_report(y_test, pred, target_names=["Non-Diabetic", "Diabetic"]))

# Sanity-check: high proba should correspond to true diabetic cases
print("--- Probability Sanity Check (first 10 test samples) ---")
print("  true | proba  | pred")
for t, p, pr in zip(y_test.values[:10], proba[:10], pred[:10]):
    print(f"   {t}   | {p:.3f}  |  {pr}")

# -----------------------------
# Save to MLflow
# -----------------------------

mlflow.set_tracking_uri("http://mlflow:5000")

with mlflow.start_run():

    mlflow.log_param("n_estimators", 500)
    mlflow.log_param("max_depth", 10)
    mlflow.log_param("min_samples_leaf", 2)
    mlflow.log_param("class_weight", "balanced")

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("roc_auc", roc_auc)

    mlflow.sklearn.log_model(
        model,
        artifact_path="model",
        registered_model_name="CDSS_Diabetes_Model"
    )

print("Diabetes model registered successfully")