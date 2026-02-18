import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# =========================
# Load Dataset
# =========================
df = pd.read_csv("data/heart.csv")

# =========================
# Separate Features & Target
# =========================
X = df.drop("target", axis=1)
y = df["target"]

# =========================
# Train-Test Split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# Train Model
# =========================
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# =========================
# Predictions
# =========================
y_pred = model.predict(X_test)

# =========================
# Evaluation
# =========================
print("===== MODEL EVALUATION =====")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# =========================
# Save Model
# =========================
joblib.dump(model, "models/heart_model.pkl")

print("\nModel saved successfully inside models/ folder!")
