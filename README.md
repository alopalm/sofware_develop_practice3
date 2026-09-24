# Iris Project: Species Classification - Practice 3

📖 **Documentation:** https://alopalm.github.io/sofware_develop_practice3/

**Authors:** Irene Ferrandez Colomer, Andrea Lopez Almela

## Overview

This project implements a small, well-structured machine learning pipeline that
classifies iris flowers into one of three species (*setosa*, *versicolor*,
*virginica*) based on four numeric measurements: sepal length, sepal width,
petal length, and petal width.

## Dataset

Iris dataset (Fisher, 1936), a classic multi-class classification dataset with
150 samples. It is available at the
[UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/53/iris)
and bundled with scikit-learn.

The CSV is **not** stored in the repository. It is generated in `data/iris.csv`
by `data/get_data.py` (see Usage), with the columns `sepal_length`,
`sepal_width`, `petal_length`, `petal_width`, `target` and `species`.

## Project structure

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
uv run jupyter lab notebooks/exploration.ipynb
```
Run all cells to regenerate the exploratory figures in `reports/figures/`
(histograms, pairplot, correlation heatmap) and, using the trained model and
the test data, the classification report and the confusion matrix.

**4. Regenerate the PDF report (optional):**
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
(description, Args, Returns). To build the HTML documentation locally:

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

## Contributing

1. Create a branch from `develop`: `git checkout -b feature/short-name`.
2. Make small, focused commits with descriptive messages.
3. Push the branch and open a Pull Request towards `develop`.
4. Once the Pull Request is reviewed, merge it. `develop` is merged into `main` for each release.

Never commit API keys, passwords, personal data, datasets or trained models.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).