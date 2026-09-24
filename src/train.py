"""Main script to execute model training and evaluate accuracies."""

from src.dataset import IrisDataset
from src.model import LogisticRegressionModel


def main():
    print("Loading and preprocessing data from CSV...")
    dataset = IrisDataset(test_size=0.25, random_state=42)
    X_train, X_test, y_train, y_test = dataset.get_data("data/iris.csv")

    print("Initializing and training the model...")
    model = LogisticRegressionModel(max_iter=200, random_state=42)
    model.train(X_train, y_train)

    # Evaluate and print accuracies
    print(f"Train accuracy: {model.model.score(X_train, y_train):.3f}")
    print(f"Test accuracy:  {model.model.score(X_test, y_test):.3f}")

    # Save the trained model
    model_path = "models/model.joblib"
    model.save(model_path)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()