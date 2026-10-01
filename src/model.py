"""Module that defines the Machine Learning model"""

import joblib
from sklearn.linear_model import LogisticRegression


class LogisticRegressionModel:
    """Clean wrapper around scikit-learn's LogisticRegression model"""

    def __init__(self, max_iter: int = 200, random_state: int = 42):
        """Initializes the LogisticRegressionModel.

        Args:
            - **max_iter** (int): Maximum number of iterations taken for the solvers to converge.
            - **random_state** (int): Seed of the pseudo random number generator.
        """
        self.model = LogisticRegression(
            max_iter=max_iter, random_state=random_state
        )

    def train(self, X_train, y_train):
        """Trains the model with the provided training data.

        Args:
            - **X_train**: Training data features.
            - **y_train**: Training data target values.
        """
        self.model.fit(X_train, y_train)

    def predict(self, X):
        """Makes predictions using the trained model.

        Args:
            - **X**: Features to predict on.

        Returns:
            - **predictions**: Predicted class labels.
        """
        return self.model.predict(X)

    def predict_proba(self, X):
        """Returns prediction probabilities.

        Args:
            - **X**: Features to compute probabilities for.

        Returns:
            - **probabilities**: Estimated probabilities for each class.
        """
        return self.model.predict_proba(X)

    def save(self, filepath: str):
        """Saves the trained model to disk using joblib.

        Args:
            - **filepath** (str): Path where the model file will be saved.
        """
        joblib.dump(self.model, filepath)

    def load(self, filepath: str):
        """Loads a trained model from disk.

        Args:
            - **filepath** (str): Path to the saved model file.
        """
        self.model = joblib.load(filepath)