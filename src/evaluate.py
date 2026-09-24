"""Evaluates the trained model and generates performance figures."""

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
CLASS_NAMES = ["setosa", "versicolor", "virginica"]


def evaluate():
    """Loads the trained model and generates the evaluation figures.

    Generates two images in reports/figures/: confusion_matrix.png and
    calibration_curves.png.
    """
    df = pd.read_csv("data/iris.csv")
    _, X_test, _, y_test = train_test_split(
        df[FEATURES], df["target"], test_size=0.25, random_state=42, stratify=df["target"]
    )

    model = joblib.load("models/model.joblib")
    y_pred = model.predict(X_test)
    proba = model.predict_proba(X_test)

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, display_labels=CLASS_NAMES, cmap="Blues", ax=ax, colorbar=False
    )
    ax.set_title("Confusion Matrix (Test Set)")
    fig.tight_layout()
    fig.savefig("reports/figures/confusion_matrix.png", dpi=150)
    print("Saved: reports/figures/confusion_matrix.png")

    # Calibration curves
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    y_test_arr = y_test.to_numpy()
    for i, (ax, name) in enumerate(zip(axes, CLASS_NAMES)):
        y_true_bin = (y_test_arr == i).astype(int)
        frac_pos, mean_pred = calibration_curve(y_true_bin, proba[:, i], n_bins=5)
        ax.plot(mean_pred, frac_pos, "o-", color="crimson", label=name)
        ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
        ax.set_title(f"Calibration Curve: {name}")
        ax.set_xlabel("Mean Predicted Probability")
        ax.set_ylabel("Fraction of Positives")
        ax.legend()
    fig.suptitle("Model Calibration Analysis")
    fig.tight_layout()
    fig.savefig("reports/figures/calibration_curves.png", dpi=150)
    print("Saved: reports/figures/calibration_curves.png")


if __name__ == "__main__":
    evaluate()