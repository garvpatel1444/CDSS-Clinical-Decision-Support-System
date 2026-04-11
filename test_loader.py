from ml_pipeline.data_loader import load_dataset

# Test Heart Dataset
X_train, X_test, y_train, y_test = load_dataset(
    "data/heart.csv",
    target_column="target"
)

print("Heart Dataset:")
print("Train size:", X_train.shape)
print("Test size:", X_test.shape)

# Test Diabetes Dataset
X_train_d, X_test_d, y_train_d, y_test_d = load_dataset(
    "data/diabetes.csv",
    target_column="Outcome"
)

print("\nDiabetes Dataset:")
print("Train size:", X_train_d.shape)
print("Test size:", X_test_d.shape)
