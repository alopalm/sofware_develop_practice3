"""Module for loading and preprocessing the dataset from CSV."""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]


class IrisDataset:
    """Class responsible for loading, preprocessing, and splitting the Iris dataset from CSV."""

    def __init__(self, test_size: float = 0.25, random_state: int = 42):
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()

    def get_data(self, filepath: str = "data/iris.csv"):
        """Loads the Iris dataset from CSV, normalizes features, and splits into train/test sets.

        Args:
            filepath (str): Path to the CSV file.

        Returns:
            X_train, X_test, y_train, y_test: Processed data ready for training.
        """
        df = pd.read_csv(filepath)
        X = df[FEATURES]
        y = df["target"]

        # Split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y,
        )

        # Normalize the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        return X_train_scaled, X_test_scaled, y_train, y_test