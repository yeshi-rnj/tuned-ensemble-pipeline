"""
Milestone: Tuned Ensemble Pipeline
ML-C1 baseline -> feature selection (ML-7) -> ensemble (ML-5) -> tuning (ML-6)

Leak-free design
  * One stratified train/test split; the test set is used exactly once, at the end.
  * Imputing, scaling, encoding and feature selection all live INSIDE sklearn
    Pipelines, so they are re-fit on each CV training fold only.
  * Hyperparameter search uses cross-validation on the training set only.

Run:  python tuned_ensemble.py
Outputs: images/*.png, results/*.csv|json|md
"""
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import randint
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, f1_score)
from sklearn.model_selection import (RandomizedSearchCV, StratifiedKFold,
                                     cross_val_score, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

# =============================== CONFIG ======================================
DATA_PATH = "data/data.csv"   # your ML-C1 dataset (CSV)
TARGET = "target"             # name of the target column
SCORING = "f1_macro"          # should match your ML-C1 metric
SEED = 42
TEST_SIZE = 0.2
N_SPLITS = 5
N_ITER = 40                   # random-search iterations
# Baseline model from ML-C1: swap in the exact estimator you used there.
BASELINE_CLF = LogisticRegression(max_iter=1000, random_state=SEED)
# =============================================================================

IMG, RES = Path("images"), Path("results")
IMG.mkdir(exist_ok=True)
RES.mkdir(exist_ok=True)


def make_preprocess(X):
    """Impute + scale numeric columns, impute + one-hot categorical columns."""
    num = X.select_dtypes(include="number").columns.tolist()
    cat = [c for c in X.columns if c not in num]
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore",
                                               sparse_output=False))]), cat),
    ])


def make_selector():
    return SelectFromModel(
        RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1),
        threshold="median")


# ------------------------------ Load & split ---------------------------------
DEMO = False
if Path(DATA_PATH).exists():
    df = pd.read_csv(DATA_PATH)
else:
    # Fallback so the script always runs. Replace with YOUR dataset before submitting!
    from sklearn.datasets import load_breast_cancer
    print("WARNING: %s not found -> using built-in DEMO dataset (breast cancer)." % DATA_PATH)
    df = load_breast_cancer(as_frame=True).frame
    TARGET, DEMO = "target", True
X, y = df.drop(columns=[TARGET]), df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED)
cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)


def cv_score(model):
    s = cross_val_score(model, X_train, y_train, cv=cv, scoring=SCORING, n_jobs=-1)
    return s.mean(), s.std()


# ------------------------------ 1. Baseline ----------------------------------
baseline = Pipeline([("prep", make_preprocess(X)), ("clf", clone(BASELINE_CLF))])
cv_results = {"1. Baseline (ML-C1)": cv_score(baseline)}

# --------------- 2. Feature selection (evidence: RF importances) -------------
prep_probe = make_preprocess(X)
Xt = prep_probe.fit_transform(X_train)          # fit on TRAIN only
feat_names = prep_probe.get_feature_names_out()
probe = RandomForestClassifier(n_estimators=500, random_state=SEED, n_jobs=-1)
probe.fit(Xt, y_train)
importances = pd.Series(probe.feature_importances_, index=feat_names
                        ).sort_values(ascending=False)
importances.to_csv(RES / "feature_importances.csv", header=["importance"])

baseline_fs = Pipeline([("prep", make_preprocess(X)), ("select", make_selector()),
                        ("clf", clone(BASELINE_CLF))])
cv_results["2. Baseline + feature selection"] = cv_score(baseline_fs)

# --------------------- 3. Ensemble on selected features ----------------------
ensemble = Pipeline([("prep", make_preprocess(X)), ("select", make_selector()),
                     ("clf", RandomForestClassifier(random_state=SEED, n_jobs=-1))])
cv_results["3. Random Forest + selection (untuned)"] = cv_score(ensemble)

# ------------------------ 4. Systematic tuning -------------------------------
param_dist = {
    "select__threshold": ["median", "mean", 0.01],
    "clf__n_estimators": randint(200, 800),
    "clf__max_depth": [None, 5, 10, 20],
    "clf__min_samples_split": randint(2, 20),
    "clf__min_samples_leaf": randint(1, 10),
    "clf__max_features": ["sqrt", "log2", 0.5],
}
search = RandomizedSearchCV(ensemble, param_dist, n_iter=N_ITER, scoring=SCORING,
                            cv=cv, random_state=SEED, n_jobs=-1, refit=True)
search.fit(X_train, y_train)
tuned = search.best_estimator_
cv_results["4. Tuned ensemble"] = (
    search.best_score_,
    search.cv_results_["std_test_score"][search.best_index_])

pd.DataFrame(search.cv_results_).sort_values("rank_test_score").to_csv(
    RES / "search_results.csv", index=False)
with open(RES / "best_params.json", "w") as f:
    json.dump({k: (v if isinstance(v, (str, int, float, type(None))) else str(v))
               for k, v in search.best_params_.items()}, f, indent=2)

# Which features survived selection in the final model?
final_names = tuned.named_steps["prep"].get_feature_names_out()
mask = tuned.named_steps["select"].get_support()
fs_table = pd.DataFrame({"feature": final_names, "kept": mask}).merge(
    importances.rename("importance"), left_on="feature", right_index=True)
fs_table.sort_values("importance", ascending=False).to_csv(
    RES / "feature_selection.csv", index=False)

# --------------------- 5. Final single test-set evaluation -------------------
baseline.fit(X_train, y_train)
rows = []
for name, m in [("Baseline (ML-C1)", baseline), ("Tuned ensemble", tuned)]:
    pred = m.predict(X_test)
    rows.append({"model": name,
                 "test_accuracy": accuracy_score(y_test, pred),
                 "test_f1_macro": f1_score(y_test, pred, average="macro")})
final = pd.DataFrame(rows)
final.to_csv(RES / "final_comparison.csv", index=False)

cv_df = pd.DataFrame(cv_results, index=[f"cv_{SCORING}_mean", "cv_std"]).T
cv_df.to_csv(RES / "cv_progression.csv")

# ================================ PICTURES ===================================
# 1) Feature importances (top 15, coloured by kept/dropped)
top = fs_table.sort_values("importance", ascending=False).head(15)[::-1]
plt.figure(figsize=(8, 6))
plt.barh(top["feature"], top["importance"],
         color=["#2a9d8f" if k else "#adb5bd" for k in top["kept"]])
plt.title("Top 15 feature importances (green = kept, grey = dropped)")
plt.xlabel("Random Forest importance")
plt.tight_layout(); plt.savefig(IMG / "feature_importances.png", dpi=150); plt.close()

# 2) CV score progression across the four stages
plt.figure(figsize=(8, 4.5))
labels = list(cv_results)
means = [v[0] for v in cv_results.values()]
stds = [v[1] for v in cv_results.values()]
bars = plt.bar(range(len(labels)), means, yerr=stds, capsize=4,
               color=["#adb5bd", "#8ecae6", "#219ebc", "#023047"])
plt.xticks(range(len(labels)), [l.split(". ")[1] for l in labels],
           rotation=15, ha="right", fontsize=8)
lo = max(0, min(means) - 0.1)
plt.ylim(lo, min(1.0, max(means) + 0.05))
for b, m in zip(bars, means):
    plt.text(b.get_x() + b.get_width() / 2, m + 0.003, f"{m:.3f}", ha="center", fontsize=8)
plt.ylabel(f"CV {SCORING} (mean ± std)")
plt.title("Cross-validation progression (train set only)")
plt.tight_layout(); plt.savefig(IMG / "cv_progression.png", dpi=150); plt.close()

# 3) Before vs after on the held-out test set
ax = final.set_index("model")[["test_accuracy", "test_f1_macro"]].plot.bar(
    figsize=(6, 4.5), rot=0, color=["#8ecae6", "#023047"])
for c in ax.containers:
    ax.bar_label(c, fmt="%.3f", fontsize=8)
ax.set_ylim(max(0, final[["test_accuracy", "test_f1_macro"]].min().min() - 0.1), 1.0)
ax.set_title("Baseline vs tuned ensemble (held-out test set)")
plt.tight_layout(); plt.savefig(IMG / "before_after.png", dpi=150); plt.close()

# 4) Confusion matrices side by side
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for a, (name, m) in zip(axes, [("Baseline", baseline), ("Tuned ensemble", tuned)]):
    ConfusionMatrixDisplay.from_estimator(m, X_test, y_test, ax=a, colorbar=False,
                                          cmap="Blues")
    a.set_title(name)
plt.tight_layout(); plt.savefig(IMG / "confusion_matrices.png", dpi=150); plt.close()

# ============================ MARKDOWN SUMMARY ===============================
imp_gain = final.loc[1, "test_f1_macro"] - final.loc[0, "test_f1_macro"]
md = ["# Results\n",
      "> DEMO RUN on a built-in sample dataset, not the real ML-C1 data.\n" if DEMO else "",
      f"Best params: `{json.dumps(search.best_params_, default=str)}`\n",
      f"Features kept: **{int(mask.sum())} / {len(mask)}**\n",
      "## Cross-validation progression (train set)\n",
      cv_df.round(4).to_markdown(), "\n",
      "## Final test-set comparison (evaluated once)\n",
      final.round(4).to_markdown(index=False), "\n",
      f"F1-macro change on test set: **{imp_gain:+.4f}**\n"]
(RES / "results.md").write_text("\n".join(md))
print("\n".join(md))
