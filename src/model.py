"""Module that defines the Machine Learning model"""

import joblib
from sklearn.linear_model import LogisticRegression


class LogisticRegressionModel:
    """Clean wrapper around scikit-learn's LogisticRegression model"""

    def __init__(self, max_iter: int = 200, random_state: int = 42):
        self.model = LogisticRegression(
            max_iter=max_iter, random_state=random_state
        )

    def train(self, X_train, y_train):
        """Trains the model with the provided training data"""
        self.model.fit(X_train, y_train)

    def predict(self, X):
        """Makes predictions using the trained model"""
        return self.model.predict(X)

    def predict_proba(self, X):
        """Returns prediction probabilities"""
        return self.model.predict_proba(X)

    def save(self, filepath: str):
        """Saves the trained model to disk using joblib"""
        joblib.dump(self.model, filepath)

    def load(self, filepath: str):
        """Loads a trained model from disk"""
        self.model = joblib.load(filepath)