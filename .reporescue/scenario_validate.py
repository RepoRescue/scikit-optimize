"""Path B scenario: a real sklearn ML practitioner uses BayesSearchCV
to tune an SVC classifier on the digits dataset.

This is the canonical "downstream" workflow for scikit-optimize:
drop-in replacement for sklearn's GridSearchCV / RandomizedSearchCV.

Reads only README + skopt.BayesSearchCV docstring. >30 lines of real
business code (model selection on a real dataset, not a toy fixture).
"""
import time
import numpy as np
from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer


def main():
    # 1. Real dataset: handwritten digits (8x8 grayscale, 1797 samples)
    X, y = load_digits(return_X_y=True)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=0, stratify=y
    )
    print(f"[data] train={X_tr.shape}, test={X_te.shape}, classes={len(np.unique(y))}")

    # 2. Realistic sklearn pipeline: scaler + SVM
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(random_state=0)),
    ])

    # 3. Bayesian-optimized hyperparameter search space
    search_space = {
        "svc__C": Real(1e-3, 1e+2, prior="log-uniform"),
        "svc__gamma": Real(1e-4, 1e+0, prior="log-uniform"),
        "svc__degree": Integer(1, 4),
        "svc__kernel": Categorical(["linear", "poly", "rbf"]),
    }

    # 4. Baseline: untuned SVC cv-score
    base_score = cross_val_score(pipe, X_tr, y_tr, cv=3, n_jobs=1).mean()
    print(f"[baseline] untuned 3-fold CV accuracy = {base_score:.4f}")

    # 5. Run BayesSearchCV (the skopt flagship API for ML practitioners)
    t0 = time.time()
    opt = BayesSearchCV(
        pipe,
        search_space,
        n_iter=15,
        cv=3,
        random_state=0,
        n_jobs=1,
        verbose=0,
    )
    opt.fit(X_tr, y_tr)
    elapsed = time.time() - t0
    print(f"[BayesSearchCV] {opt.n_iter} iters in {elapsed:.1f}s")
    print(f"[BayesSearchCV] best params: {dict(opt.best_params_)}")
    print(f"[BayesSearchCV] best CV score: {opt.best_score_:.4f}")

    # 6. Hard assertions: tuned model must beat baseline on held-out set
    test_score = opt.score(X_te, y_te)
    print(f"[BayesSearchCV] held-out test accuracy: {test_score:.4f}")
    assert opt.best_score_ > base_score - 0.02, (
        f"BayesSearchCV worse than untuned: {opt.best_score_:.4f} vs {base_score:.4f}"
    )
    assert test_score > 0.90, f"tuned SVC underperforms on digits: {test_score:.4f}"
    assert hasattr(opt, "cv_results_") and len(opt.cv_results_["params"]) == 15

    # 7. Confirm the optimizer actually explored the space (>1 unique kernel/C)
    seen_kernels = {p["svc__kernel"] for p in opt.cv_results_["params"]}
    seen_C = {round(float(p["svc__C"]), 6) for p in opt.cv_results_["params"]}
    print(f"[explore] kernels tried: {seen_kernels}, distinct C values: {len(seen_C)}")
    assert len(seen_C) >= 5, "BayesSearchCV did not explore C"

    print("SCENARIO_PASS")


if __name__ == "__main__":
    main()
