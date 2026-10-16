# Iris Project — Species Classification - Pract 4

📖 **Documentation:** https://alopalm.github.io/sofware_develop_practice3/

**Authors:** Irene Ferrandez Colomer, Andrea Lopez Almela

## Overview

This project implements a small, well-structured machine learning pipeline that
classifies iris flowers into one of three species (*setosa*, *versicolor*,
*virginica*) based on four numeric measurements: sepal length, sepal width,
petal length, and petal width. It also includes an interactive web application built with **Gradios** for data exploration, model training, and evaluation.


## Dataset

We use the **Iris dataset**, a classic multi-class classification dataset with
150 samples.
You can check the origin and details of the dataset at the official repository:
(https://archive.ics.uci.edu/dataset/53/iris)





The CSV is **not** stored in the repository. It is generated in `data/iris.csv`
by `data/get_data.py` (see Usage), with the columns `sepal_length`,
`sepal_width`, `petal_length`, `petal_width`, `target` and `species`.

- `app.py` — Main Gradio interactive web application
- `requirements.txt` — Project runtime dependencies for deployment
- `data/` — dataset download script (`get_data.py`) and the generated `iris.csv`
- `src/` — source code: `train.py` (training pipeline) and `evaluate.py` (evaluation pipeline)
- `notebooks/` — `exploration.ipynb`, the exploratory data analysis notebook
- `models/` — trained model artifact (`model.joblib`), generated after running `train.py`
- `reports/` — `report.md` / `report.pdf` (performance report) and `figures/` (all generated PNG charts)
- `docs/` — HTML documentation generated automatically by pdoc
- `pyproject.toml` and `uv.lock` — project dependencies, managed with uv
- `README.md` — this file

- `data/`: dataset download script (`get_data.py`) and the generated `iris.csv` (not versioned)
- `src/`: source code
  - `dataset.py`: `IrisDataset`, loads, preprocesses and splits the data
  - `model.py`: `LogisticRegressionModel`, wrapper around scikit-learn's `LogisticRegression`
  - `train.py`: training pipeline that connects dataset and model
- `notebooks/`: `exploration.ipynb`, exploratory data analysis and evaluation of the trained model
- `models/`: trained model artifact (`model.joblib`), generated after running the training (not versioned)
- `reports/`: `report.md` / `report.pdf` (performance report) and `figures/` (generated PNG charts)
- `.github/workflows/docs.yml`: publishes the documentation with GitHub Pages
- `pyproject.toml` and `uv.lock`: project dependencies, managed with uv
- `LICENSE`: MIT license
- `README.md`: this file

## Installation

Clone the repository and, from its root folder, run:

```
git clone https://github.com/alopalm/sofware_develop_practice3.git
cd sofware_develop_practice3
uv sync
```

This creates a `.venv/` folder and installs every dependency listed in
`pyproject.toml` (both the runtime dependencies like `pandas` and
`scikit-learn`, and the development dependencies like `pdoc` and `jupyter`).
Alternatively, if you are deploying or running the web application directly, you can install dependencies using:

```
pip install -r requirements.txt
```

## Usage

Run each step from the project root, in this order:

**1. Download / generate the dataset:**
```
uv run python data/get_data.py
```
This saves `data/iris.csv`. If the file is missing, the training step
generates it automatically.

**2. Train the model:**
```
uv run python -m src.train
```
This trains a logistic regression classifier and saves it to
`models/model.joblib`.

**3. Explore the data and evaluate the model:**
```
uv run python src/evaluate.py
```
This generates `reports/figures/confusion_matrix.png` and
`reports/figures/calibration_curves.png`.

**4. Run the Interactive Gradio Web Application:**
```
python app.py
```
This launches a local web server featuring three interactive tabs:
- **Data Exploration:** View descriptive statistics and interactive feature scatter plots.
- **Training Interface:** Adjust hyperparameters (test size, max iterations) and train the model dynamically.
- **Model Evaluation:** Evaluate the trained model performance and display the confusion matrix.

**5. Explore the dataset (optional, for EDA):**
```
uv run jupyter notebook notebooks/exploration.ipynb
```
Run all cells to regenerate the exploratory figures in `reports/figures/`
(histograms, pairplot, correlation heatmap) and, using the trained model and
the test data, the classification report and the confusion matrix.

**6. Regenerate the PDF report (optional):**
```
cd reports
pandoc report.md -o report.pdf --pdf-engine=wkhtmltopdf
cd ..
```

## Documentation

The documentation is published online and updated automatically by GitHub
Actions every time a change to `src/`, `docs/` or `README.md` is pushed to `main`:

**https://alopalm.github.io/sofware_develop_practice3/**

Every module and function is documented with Google-style docstrings
(description, Args, Returns). Browsable HTML documentation is generated
from these docstrings using pdoc.
You can check the complete technical documentation of the code generated with pdoc at the following link:
(https://alopalm.github.io/sofware_develop_practice3/)

```
uv run pdoc src -o docs_html --docformat google
```

Then open `docs_html/index.html` in a browser.

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

## Docker Deployment

This project includes a Docker container to easily run the interactive demonstration without needing to manually configure the environment.

### Prerequisites
* Make sure you have [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed on your system.

### Run the container from Docker Hub
You can pull and run the latest image directly from Docker Hub:

```bash
docker pull ireneferrandez/sofware_develop_practice3:latest
docker run -p 7860:7860 ireneferrandez/sofware_develop_practice3:latest
