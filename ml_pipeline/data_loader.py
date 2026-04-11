import pandas as pd
from sklearn.model_selection import train_test_split


def load_dataset(file_path, target_column, test_size=0.2, random_state=42):
    """
    Generic dataset loader for any disease dataset.
    """

    # Load CSV
    df = pd.read_csv(file_path)

    # Separate features & target
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y  # important for medical datasets
    )

    return X_train, X_test, y_train, y_test
