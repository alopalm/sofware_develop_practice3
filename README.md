# Iris Project — Species Classification - Pract 3

**Authors:** 
Irene Ferrandez Colomer
Andrea Lopez Almela


## Overview

This project implements a small, well-structured machine learning pipeline that
classifies iris flowers into one of three species (*setosa*, *versicolor*,
*virginica*) based on four numeric measurements: sepal length, sepal width,
petal length, and petal width.


## Dataset

We use the **Iris dataset**, a classic multi-class classification dataset with
150 samples.

## Project structure

- `data/` — dataset download script (`get_data.py`) and the generated `iris.csv`
- `src/` — source code: `train.py` (training pipeline) and `evaluate.py` (evaluation pipeline)
- `notebooks/` — `exploration.ipynb`, the exploratory data analysis notebook
- `models/` — trained model artifact (`model.joblib`), generated after running `train.py`
- `reports/` — `report.md` / `report.pdf` (performance report) and `figures/` (all generated PNG charts)
- `docs/` — HTML documentation generated automatically by pdoc
- `pyproject.toml` and `uv.lock` — project dependencies, managed with uv
- `README.md` — this file


## Installation

Clone or unzip this project, then from its root folder run:

```
uv sync
```

This creates a `.venv/` folder and installs every dependency listed in
`pyproject.toml` (both the runtime dependencies like `pandas` and
`scikit-learn`, and the development dependencies like `pdoc` and `jupyter`).

## Usage

Run each step from the project root, in this order:

**1. Download / generate the dataset:**
```
uv run python data/get_data.py
```
This saves `data/iris.csv`.

**2. Train the model:**
```
uv run python src/train.py
```
This trains a logistic regression classifier and saves it to
`models/model.joblib`. It also prints the train/test accuracy to the console.

**3. Evaluate the model and generate performance figures:**
```
uv run python src/evaluate.py
```
This generates `reports/figures/confusion_matrix.png` and
`reports/figures/calibration_curves.png`.

**4. Explore the dataset (optional, for EDA):**
```
uv run jupyter notebook notebooks/exploration.ipynb
```
Run all cells to regenerate the exploratory figures in `reports/figures/`
(histograms, pairplot, correlation heatmap).

**5. Regenerate the PDF report (optional):**
```
cd reports
pandoc report.md -o report.pdf --pdf-engine=wkhtmltopdf
cd ..
```

## Documentation

Every module and function is documented with Google-style docstrings
(description, Args, Returns). Browsable HTML documentation is generated
from these docstrings using pdoc:

```
uv run pdoc src/train.py src/evaluate.py -o docs/ --docformat google
```

Open `docs/index.html` in a browser to navigate the documentation.

## Dependency management

This project uses **uv** for dependency management and virtual environments
(see `pyproject.toml` and `uv.lock`). Runtime dependencies are pinned to a
major version range (e.g. `pandas>=2.2,<3`) so that `uv sync` on a fresh
machine always reproduces a working environment, while still allowing minor
and patch updates. Development-only dependencies (`pdoc`, `jupyter`) are kept
in a separate `dependency-groups.dev` section, since they are needed to
develop and document the project but not to run the trained model in
production.

## Results summary

The logistic regression model achieves **94.7% accuracy** on the held-out test
set (25% of the data, stratified split). The main source of error is confusion
between *versicolor* and *virginica*, the two most visually similar species,
as shown both in the exploratory pairplot and in the confusion matrix. Full
details and figure interpretations are available in `reports/report.pdf`.