# Bayesian Logistic Regression for Heart Disease Risk

## 1. Objective

To model the probability of heart disease using interpretable Bayesian logistic regression and quantify predictive and parameter uncertainty.

---

## 2. Data

Dataset: UCI Heart Disease  
Binary target defined as num > 0  

Train/test split: 80/20 stratified  

---

## 3. Model Specification

Logistic regression:

logit(P(Y=1|X)) = beta_0 + X beta

Priors:

- beta_0 ~ Normal(0, 1)
- beta_j ~ Normal(0, 1)

Inference method:

- NUTS sampler
- 1500 tuning steps
- 1500 posterior draws
- 2 chains

---

## 4. Discrimination Performance

Posterior ROC AUC:

Mean: 0.897  
95 percent credible interval: 0.880 to 0.913  

Interpretation:

The model demonstrates strong ability to rank patients by disease risk.

---

## 5. Calibration

Brier score: 0.1218  

Calibration curve shows approximate alignment with the identity line.

Interpretation:

Predicted probabilities are numerically reliable and not systematically biased.

---

## 6. Posterior Odds Ratios

Robust predictors (credible interval excludes 1):

- Number of major vessels (ca)
- ST depression (oldpeak)
- Age

Less stable predictors:

- Several categorical features show wide credible intervals.

Interpretation:

Continuous clinical measurements carry the strongest and most stable signals.

---

## 7. Comparison with Frequentist Approach

Bootstrap frequentist logistic regression produced similar AUC and effect directions.

Agreement between bootstrap confidence intervals and Bayesian credible intervals suggests stable inference.

---

## 8. Conclusion

The model provides:

- Strong discrimination
- Good calibration
- Interpretable effect estimates
- Full posterior uncertainty quantification

This framework is appropriate for healthcare risk modeling where probabilistic interpretation is essential.
