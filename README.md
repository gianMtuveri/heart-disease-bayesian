# Bayesian Heart Disease Risk Modeling

## Overview

This repository implements Bayesian logistic regression to model heart disease risk using the UCI Heart Disease dataset.

The project emphasizes:

- Interpretable modeling
- Posterior uncertainty quantification
- Predictive performance assessment
- Calibration analysis

Unlike frequentist logistic regression, this model estimates a full posterior distribution over parameters and predictions.

---

## Dataset

Source: UCI Heart Disease dataset.

Binary target definition:

- 0 = No heart disease
- 1 = Presence of heart disease (num > 0)

---

## Methodology

### Preprocessing

Features are grouped by type using a ColumnTransformer:

Numeric features  
- Median imputation  
- Standardization  

Boolean features  
- Most frequent imputation  
- Converted to 0/1  

Categorical features  
- Most frequent imputation  
- One-hot encoding  

All transformations are fitted on the training set only.

---

### Bayesian Logistic Regression

Model:

logit(P(Y=1|X)) = intercept + X * beta

Priors:

- intercept ~ Normal(0, 1)
- beta_j ~ Normal(0, 1)

Posterior inference performed using NUTS (Hamiltonian Monte Carlo) via PyMC.

---

## Results Summary

Test set evaluation:

Posterior ROC AUC  
Mean: approximately 0.897  
95 percent credible interval: approximately 0.880 to 0.913  

Classification accuracy: approximately 0.84  

Calibration:

Brier score: 0.1218  

This indicates:

- Strong discrimination
- Good probability calibration
- Stable predictive performance

---

## Posterior Odds Ratios

Odds ratios are computed as exp(beta).

Strong and stable predictors include:

- Number of major vessels (ca)
- ST depression (oldpeak)
- Age

Categorical predictors show wider credible intervals, indicating greater uncertainty.

---

## Outputs

Generated artifacts:

results/bayes_auc_draws.csv  
results/bayes_odds_ratios.csv  
results/fig_auc_posterior.png  
results/fig_or_forest.png  
results/fig_calibration.png  

---

## How to Run

1) Install dependencies

pip install -r requirements.txt

2) Place dataset at:

data/heart_disease_uci.csv

3) Run

python src/bayes_logit.py

---

## License

Educational and portfolio use.


