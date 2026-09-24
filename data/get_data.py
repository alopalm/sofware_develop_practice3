"""Download the Iris dataset and save it as a CSV file."""
from sklearn.datasets import load_iris
import pandas as pd

d = load_iris(as_frame=True)
df = d.frame
df.columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "target"]
df["species"] = df["target"].map(dict(enumerate(d.target_names)))
df.to_csv("data/iris.csv", index=False)
print(f"Saved data/iris.csv with {df.shape[0]} rows")