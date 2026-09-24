"""Carga, preprocesado y partición del dataset Iris."""
from pathlib import Path

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
TARGET = "target"


def create_csv(filepath: str) -> None:
    """Genera el CSV de Iris con las columnas del proyecto si no existe."""
    iris = load_iris(as_frame=True)
    df = iris.frame.rename(columns=dict(zip(iris.feature_names, FEATURES)))
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)


class IrisDataset:
    """Carga, preprocesa y divide el dataset Iris a partir de un CSV."""

    def __init__(self, test_size: float = 0.25, random_state: int = 42):
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()

    def load(self, filepath: str = "data/iris.csv"):
        """Lee el CSV y devuelve las características X y la etiqueta y.

        Si el fichero no existe, lo genera a partir de scikit-learn.
        """
        if not Path(filepath).exists():
            create_csv(filepath)
        df = pd.read_csv(filepath)
        return df[FEATURES], df[TARGET]

    def preprocess(self, X_train, X_test):
        """Estandariza las características (ajusta solo con train)."""
        return self.scaler.fit_transform(X_train), self.scaler.transform(X_test)

    def get_data(self, filepath: str = "data/iris.csv"):
        """Devuelve X_train, X_test (escalados), y_train, y_test."""
        X, y = self.load(filepath)
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y,
        )
        X_train, X_test = self.preprocess(X_train, X_test)
        return X_train, X_test, y_train, y_test