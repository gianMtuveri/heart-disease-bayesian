# src/bayes_logit.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, classification_report, brier_score_loss
from sklearn.calibration import calibration_curve
import pymc as pm


DATA_PATH = "data/heart_disease_uci.csv"

def load_data(path: str) -> pd.DataFrame:
    # parse TRUE/FALSE-like tokens robustly
    return pd.read_csv(
        path,
        true_values=["TRUE", "True", "true"],
        false_values=["FALSE", "False", "false"],
    )


def add_target_and_drop_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["target"] = (df["num"] > 0).astype(int)

    drop_cols = ["num"]
    if "id" in df.columns:
        drop_cols.append("id")
    if "dataset" in df.columns:
        drop_cols.append("dataset")

    return df.drop(columns=drop_cols)


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    bool_cols = X.select_dtypes(include=["bool", "boolean"]).columns.tolist()
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    bool_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("to_int", FunctionTransformer(lambda a: a.astype(int))),
    ])

    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_cols),
            ("bool", bool_pipe, bool_cols),
            ("cat", cat_pipe, categorical_cols),
        ],
        remainder="drop",
    )
    return pre


def to_dense(a):
    # OneHotEncoder often returns a sparse matrix; PyMC wants dense numpy.
    return a.toarray() if hasattr(a, "toarray") else np.asarray(a)


def fit_bayesian_logit(X_train_t, y_train, draws, tune, chains, target_accept=0.9, seed=42):
    """
    Bayesian logistic regression with regularizing priors:
      beta_j ~ Normal(0, 1)
      intercept ~ Normal(0, 1)

    Because numeric features are standardized and one-hot features are 0/1,
    Normal(0,1) is a reasonable shrinkage prior.
    """
    X_train_t = np.asarray(X_train_t, dtype=float)
    y_train = np.asarray(y_train, dtype=int)

    n, p = X_train_t.shape

    with pm.Model() as model:
        intercept = pm.Normal("intercept", mu=0.0, sigma=1.0)
        beta = pm.Normal("beta", mu=0.0, sigma=1.0, shape=p)

        logits = intercept + pm.math.dot(X_train_t, beta)
        pm.Bernoulli("y", logit_p=logits, observed=y_train)

        trace = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            random_seed=seed,
            target_accept=target_accept,
            progressbar=True,
        )

    return trace


def posterior_predict_proba(trace, X):
    """
    Compute posterior predictive probabilities on X for each posterior draw.
    Returns: proba_draws with shape (n_draws, n_obs)
    """
    X = np.asarray(X, dtype=float)

    # Extract samples and flatten chains
    intercept = trace.posterior["intercept"].values  # (chains, draws)
    beta = trace.posterior["beta"].values           # (chains, draws, p)

    intercept = intercept.reshape(-1)               # (n_draws_total,)
    beta = beta.reshape(-1, beta.shape[-1])         # (n_draws_total, p)

    logits = intercept[:, None] + beta @ X.T        # (n_draws_total, n_obs)
    proba = 1.0 / (1.0 + np.exp(-logits))
    return proba


def summarize_auc(proba_draws, y_test):
    y_test = np.asarray(y_test, dtype=int)

    aucs = np.array([roc_auc_score(y_test, proba_draws[i]) for i in range(proba_draws.shape[0])])

    mean = float(aucs.mean())
    std = float(aucs.std(ddof=1))
    ci_low, ci_high = np.percentile(aucs, [2.5, 97.5])

    print("\n=== BAYESIAN ROC AUC (POSTERIOR) ===")
    print(f"Mean : {mean:.4f}")
    print(f"Std  : {std:.4f}")
    print(f"95% CrI: ({ci_low:.4f}, {ci_high:.4f})")

    return aucs


def summarize_odds_ratios(trace, feature_names):
    """
    Posterior odds ratios:
      OR_j = exp(beta_j)

    Returns a DataFrame with mean OR and 95% credible interval.
    """
    beta = trace.posterior["beta"].values  # (chains, draws, p)
    beta = beta.reshape(-1, beta.shape[-1])  # (n_draws_total, p)

    or_samples = np.exp(beta)  # (n_draws_total, p)

    rows = []
    for j, name in enumerate(feature_names):
        s = or_samples[:, j]
        mean = float(s.mean())
        ci_low, ci_high = np.percentile(s, [2.5, 97.5])
        rows.append({
            "feature": name,
            "odds_ratio_mean": mean,
            "ci_low": float(ci_low),
            "ci_high": float(ci_high),
        })

    df = pd.DataFrame(rows).sort_values("odds_ratio_mean", ascending=False)
    return df


def plot_auc_posterior(
    auc_csv_path: str = "results/bayes_auc_draws.csv",
    out_path: str = "results/fig_auc_posterior.png",
    bins: int = 40,
) -> None:
    """
    Plot posterior distribution of ROC AUC (histogram) with mean and 95% credible interval.
    Input: CSV with column 'bayes_auc'
    Output: PNG saved to out_path
    """
    df = pd.read_csv(auc_csv_path)
    if "bayes_auc" not in df.columns:
        raise ValueError(f"Expected column 'bayes_auc' in {auc_csv_path}. Found: {list(df.columns)}")

    auc = df["bayes_auc"].to_numpy(dtype=float)

    mean = float(np.mean(auc))
    ci_low, ci_high = np.percentile(auc, [2.5, 97.5])

    plt.figure(figsize=(8, 5))
    plt.hist(auc, bins=bins, edgecolor="black")
    plt.axvline(mean, linewidth=2, label=f"Mean = {mean:.4f}")
    plt.axvline(ci_low, linestyle="--", linewidth=2, label=f"2.5% = {ci_low:.4f}")
    plt.axvline(ci_high, linestyle="--", linewidth=2, label=f"97.5% = {ci_high:.4f}")

    plt.title("Posterior distribution of ROC AUC (test set)")
    plt.xlabel("ROC AUC")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"Saved: {out_path}")


def plot_odds_ratio_forest(
    or_csv_path: str = "results/bayes_odds_ratios.csv",
    out_path: str = "results/fig_or_forest.png",
    top_n: int = 15,
    sort_by: str = "odds_ratio_mean",
) -> None:
    """
    Plot a forest plot of posterior odds ratios with 95% credible intervals.

    Input CSV columns (expected):
      - feature
      - odds_ratio_mean
      - ci_low
      - ci_high

    Notes:
    - Uses log scale for x-axis (standard for OR plots).
    - Draws a vertical reference line at OR = 1.
    - Shows top_n features by sort_by (default: odds_ratio_mean).
    """
    df = pd.read_csv(or_csv_path)

    required = {"feature", "odds_ratio_mean", "ci_low", "ci_high"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {or_csv_path}: {missing}. Found: {list(df.columns)}")

    df = df.copy()
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["odds_ratio_mean", "ci_low", "ci_high"])

    if sort_by not in df.columns:
        raise ValueError(f"sort_by='{sort_by}' not in columns: {list(df.columns)}")

    df = df.sort_values(sort_by, ascending=False).head(top_n)

    # Reverse so the biggest is at the top in the plot
    df = df.iloc[::-1].reset_index(drop=True)

    y = np.arange(len(df))
    x = df["odds_ratio_mean"].to_numpy(dtype=float)
    x_low = df["ci_low"].to_numpy(dtype=float)
    x_high = df["ci_high"].to_numpy(dtype=float)

    # Error bars: distance from mean to interval bounds
    xerr = np.vstack([x - x_low, x_high - x])

    plt.figure(figsize=(9, 0.45 * len(df) + 2.5))
    plt.errorbar(
        x,
        y,
        xerr=xerr,
        fmt="o",
        capsize=3,
        elinewidth=2,
    )
    plt.axvline(1.0, linestyle="--", linewidth=2)

    plt.yticks(y, df["feature"])
    plt.xscale("log")
    plt.xlabel("Odds ratio (log scale)")
    plt.title(f"Posterior odds ratios (mean and 95% credible interval) — Top {top_n}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"Saved: {out_path}")


def plot_calibration(y_test, proba_mean, out_path="results/fig_calibration.png"):
    """
    Plot calibration curve and compute Brier score.
    """

    prob_true, prob_pred = calibration_curve(
        y_test,
        proba_mean,
        n_bins=10,
        strategy="quantile"
    )

    brier = brier_score_loss(y_test, proba_mean)

    plt.figure(figsize=(6, 6))
    plt.plot(prob_pred, prob_true, marker="o")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed frequency")
    plt.title(f"Calibration curve (Brier = {brier:.4f})")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"Saved: {out_path}")
    print(f"Brier score: {brier:.4f}")


def main():
    df = load_data(DATA_PATH)
    df = add_target_and_drop_cols(df)

    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    pre = build_preprocessor(X_train)
    X_train_t = pre.fit_transform(X_train)
    X_test_t = pre.transform(X_test)

    X_train_t = to_dense(X_train_t)
    X_test_t = to_dense(X_test_t)

    feature_names = pre.get_feature_names_out()

    # Fit Bayesian logistic regression
    trace = fit_bayesian_logit(X_train_t, y_train, draws=1500, tune=1500, chains=2, target_accept=0.9)

    # Posterior predictive probabilities on test
    proba_draws = posterior_predict_proba(trace, X_test_t)

    # Posterior distribution of AUC
    aucs = summarize_auc(proba_draws, y_test)

    # Point predictions from posterior mean probability (for a report-like classification table)
    proba_mean = proba_draws.mean(axis=0)
    plot_calibration(y_test, proba_mean)
    preds = (proba_mean >= 0.5).astype(int)

    print("\n=== CLASSIFICATION REPORT (POSTERIOR MEAN THRESHOLD 0.5) ===")
    print(classification_report(y_test, preds, digits=3))

    # Posterior odds ratios
    or_df = summarize_odds_ratios(trace, feature_names)

    # Save artifacts
    os.makedirs("results", exist_ok=True)
    pd.DataFrame({"bayes_auc": aucs}).to_csv("results/bayes_auc_draws.csv", index=False)
    or_df.to_csv("results/bayes_odds_ratios.csv", index=False)

    print("\nSaved results/bayes_auc_draws.csv")
    print("Saved results/bayes_odds_ratios.csv")
    print("\nTop Odds Ratios (posterior mean):")
    print(or_df.head(10))


    plot_auc_posterior(
        auc_csv_path="results/bayes_auc_draws.csv",
        out_path="results/fig_auc_posterior.png",
    )

    plot_odds_ratio_forest(
        or_csv_path="results/bayes_odds_ratios.csv",
        out_path="results/fig_or_forest.png",
        top_n=15,
    )

if __name__ == "__main__":
    main()
