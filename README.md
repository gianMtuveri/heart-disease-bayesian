# Bayesian Heart Disease Risk Modeling

## Overview

This repository implements Bayesian logistic regression to model heart disease risk using the [UCI Heart Disease dataset](https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data).

The key difference from standard logistic regression is that Bayesian inference estimates a full posterior distribution for model parameters. This enables uncertainty quantification for:

- Predictive performance (posterior distribution of ROC AUC)
- Feature effects (posterior odds ratios with credible intervals)
- Predicted risk (posterior predictive probabilities)

The focus is interpretable inference and uncertainty, not complex deep learning models.

---

## Key Outputs

Running the main script produces:

- results/bayes_auc_draws.csv
  Posterior draws of ROC AUC evaluated on the fixed test set

- results/bayes_odds_ratios.csv
  Posterior odds ratios for each feature with 95% credible intervals

Optional figures (if you include the plotting functions):

- results/fig_auc_posterior.png
  Posterior distribution of AUC (histogram) with mean and 95% interval

- results/fig_or_forest.png
  Forest plot of odds ratios (log scale) with 95% intervals and OR = 1 reference line

---

## Repository Structure

Expected structure:

    heart-disease-bayesian/
    ├── src/
    │   └── bayes_logit.py
    ├── data/
    │   └── heart_disease_uci.csv        (not committed)
    ├── results/
    │   ├── bayes_auc_draws.csv
    │   ├── bayes_odds_ratios.csv
    │   ├── fig_auc_posterior.png        (optional)
    │   └── fig_or_forest.png            (optional)
    ├── requirements.txt
    ├── .gitignore
    └── README.md

---

## Dataset

Dataset: UCI Heart Disease dataset.

Expected path:

    data/heart_disease_uci.csv

Binary target definition used in this project:

- target = 0 when num == 0
- target = 1 when num > 0

The script drops non-feature columns such as id, dataset, and num (after creating target).

---

## Preprocessing

Preprocessing is implemented with scikit-learn ColumnTransformer to avoid leakage and keep transformations consistent between train and test.

Feature groups:

1) Numeric features
- Missing values: median imputation
- Scaling: standardization (mean 0, standard deviation 1) based on training set only

2) Boolean features
- Missing values: most frequent imputation
- Conversion: True/False mapped to 1/0

3) Categorical features
- Missing values: most frequent imputation
- Encoding: one-hot encoding with handle_unknown="ignore"

After preprocessing, the design matrix is fully numeric.

---

## Bayesian Logistic Regression Model

Model form:

- Linear predictor: intercept + X times beta
- Probability: logistic function of the linear predictor
- Likelihood: Bernoulli outcomes

Priors (regularizing):

- intercept is Normal(0, 1)
- each beta coefficient is Normal(0, 1)

These priors shrink extreme coefficients and improve stability, especially with correlated predictors or limited sample size.

Sampling:

- NUTS (Hamiltonian Monte Carlo) using PyMC
- Typical run uses tune, draws, and chains (for example 1500 tune, 1500 draws, 2 chains)

Notes on interpretation:

- For standardized numeric features, "one unit" corresponds to one standard deviation of the original feature.
- For one-hot encoded features, the coefficient represents the effect of that category relative to the implicit baseline induced by encoding and regularization.

---

## Posterior Predictive Evaluation

Train and test split:

- Stratified split to preserve class proportions
- Test set is held fixed during evaluation

Posterior predictive probabilities:

- For each posterior draw of parameters, compute predicted probabilities on the test set
- This yields a distribution of probabilities per test example

Posterior AUC:

- Compute ROC AUC on the test set for each posterior draw
- Save the AUC draws to results/bayes_auc_draws.csv
- Report mean and 95% credible interval from this distribution

Classification report:

- Use posterior mean probability per test example
- Convert to class label using threshold 0.5
- Report precision, recall, f1-score, and accuracy

---

## Posterior Odds Ratios

Odds ratio is computed as:

- odds_ratio = exp(beta)

For each feature:

- Compute odds ratios across posterior draws
- Summarize with:
  - odds_ratio_mean
  - 95% credible interval (2.5th and 97.5th percentiles)

Saved to:

- results/bayes_odds_ratios.csv

Columns:

- feature
- odds_ratio_mean
- ci_low
- ci_high

Interpretation:

- odds_ratio_mean greater than 1 suggests increased odds of disease
- odds_ratio_mean less than 1 suggests decreased odds of disease
- if the credible interval includes 1, the direction or magnitude is not strongly identified by the data under this model

---

## How to Run

1) Create and activate a virtual environment (recommended)

    python3 -m venv .venv
    source .venv/bin/activate

2) Install dependencies

    pip install -r requirements.txt

3) Place dataset at:

    data/heart_disease_uci.csv

4) Run Bayesian analysis

    python src/bayes_logit.py

---

## Performance Notes (PyMC and compilation)

Bayesian sampling benefits strongly from a working C compiler and Python headers.

On Amazon Linux / CloudShell, install:

    sudo yum install gcc-c++ python3-devel

If compilation is not available, sampling may still run but can be much slower.

---

## License

Educational and portfolio use.

Dataset credit: UCI Machine Learning Repository.
