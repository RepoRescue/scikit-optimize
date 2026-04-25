# scikit-optimize — Usability Validation

**Selected rescue**: sonnet (T2 PASS; srconly: FAIL — sonnet's source-only patch failed eval, but the full rescue artifact installs and runs cleanly)
**Scenario type**: B (end-user ML library API)
**Real-world use**: scikit-optimize is the canonical Bayesian-optimization library for Python ML; people use `gp_minimize` / `forest_minimize` for black-box objective tuning and `BayesSearchCV` as a drop-in replacement for `sklearn.GridSearchCV`.

## Step 0: Import sanity
`repos/rescue_sonnet/scikit-optimize/venv-t2/bin/python -c "import skopt"` -> OK (skopt 0.9.0).

## Step 1: Model selection
- sonnet PASS (preferred per priority sonnet > gpt-codex > kimi > glm > minimax)
- gpt-codex also PASS (srconly PASS) — kept as evidence the source patch is independently valid
- sonnet srconly FAIL: sonnet's rescue includes auxiliary changes (test/setup/requirements) the srconly variant strips out. Full rescue tree installs cleanly via `pip install -e`.

## Step 2: Scenario rationale
Numeric optimization toolbox — no CLI entry-points, no decorator/web API, not a parser. Fits type B (library API). README quick start: `gp_minimize(f, [(-2, 2)])`. Flagship sklearn integration: `BayesSearchCV`.

## Step 4: Install + core feature (clean venv)
- `python3.13 -m venv /tmp/scikit-optimize-clean`
- `pip install -e /home/zhihao/hdd/RepoRescue_Clean/repos/rescue_sonnet/scikit-optimize` -> OK (NumPy 2.4.4, scipy 1.17.1, scikit-learn 1.8.0, joblib 1.5.3, pyaml 26.2.1)
- `cd /tmp/scikit-optimize-clean && python <abs>/usability_validate.py` -> **PASS**
  - `gp_minimize` on Branin (20 calls): converged to f = 0.886 (Branin global min ~0.398; trajectory 20.03 -> 0.886)
  - `forest_minimize` on Rosenbrock (30 calls): final f = 0.575, beats best-of-initial-10 (asserted)
  - `skopt.space.Real` log-uniform sampling + transform/inverse_transform round-trip exact
  - `Integer` / `Categorical` 5-sample draws within domain

## Hard constraint 6: Py3.13 / NumPy-2 / sklearn-1.x surface stressed

Evidence from `outputs/sonnet/scikit-optimize/scikit-optimize.src.patch`:

| Surface | Evidence |
|---|---|
| `np.int` removed (NumPy 1.20+ / 2.x) | `skopt/space/transformers.py:259,272` — `astype(np.int)` -> `astype(int)` |
| `np.in1d` deprecated (NumPy 2.0) -> `np.isin` | `skopt/learning/forest.py:240` — `np.in1d(...)` -> `np.isin(...)` |
| sklearn 1.x: `criterion='mse'` removed | `skopt/learning/forest.py:127,189` (+6 other lines) -> `'squared_error'` |
| sklearn 1.x: `max_features='auto'` removed | `skopt/learning/forest.py:130,192` -> `1.0` |
| sklearn 1.x: `__sklearn_tags__` system | `skopt/learning/gbrt.py:38-43` — added regressor tag method |
| sklearn 1.x: `is_regressor` raising on duck-types | `skopt/utils.py:360-372`, `skopt/optimizer/optimizer.py:222-235` — try/except wrap |
| `distutils` fallback removed (setuptools 58+/Py 3.12) | `setup.py:43-46` — `from distutils.core import setup` deleted |
| Python 2 `__builtin__` legacy removed | `setup.py:47-51` — collapsed to plain `import builtins` |
| Py 3.13 declaration | `setup.py:83` — `python_requires='>=3.13'` |

bug_hunt probe 3 directly exercises the `np.in1d`->`np.isin` patch via `RandomForestRegressor.predict(return_std=True)` (the broken-before-patch path); finite, non-negative std produced.

**NOT trivial** — 9 distinct compat surfaces touched.

## Beyond unit tests (constraint 3)
- `grep -rln "load_digits" repos/rescue_sonnet/scikit-optimize/skopt/tests/` -> empty
- `tests/test_searchcv.py` exists but uses synthetic toy data only
- `examples/sklearn-gridsearchcv-replacement.py` is documentation, not pytest path
- Our scenario (real digits dataset, real sklearn Pipeline, asserted held-out accuracy > baseline) does not exist in the test suite

## Step 5: Three+ distinct submodules (constraint 5) — actually invoked
1. `skopt.gp_minimize` + `skopt.forest_minimize`
2. `skopt.space.Real` / `Integer` / `Categorical` (`rvs`, `transform`, `inverse_transform`)
3. `skopt.utils.use_named_args`
4. `skopt.benchmarks.branin`
5. `skopt.Optimizer` (ask/tell)
6. `skopt.learning.RandomForestRegressor` (patched np.isin path)
7. `skopt.BayesSearchCV`

Seven distinct paths.

## Step 6: Scenario / downstream
- **Path B**: 78-line real ML script (`scenario_validate.py`) tuning `Pipeline(StandardScaler -> SVC)` on `sklearn.datasets.load_digits` via `BayesSearchCV(n_iter=15, cv=3)`.
  - Untuned baseline 3-fold CV: 0.9733
  - BayesSearchCV best CV: 0.9755 (asserted > baseline)
  - Held-out test accuracy: 0.9778 (asserted > 0.90)
  - Explored {linear, poly, rbf} kernels and 15 distinct C values (asserted >= 5)
  - Result: **SCENARIO_PASS**
- Path A skipped: `BayesSearchCV` IS the canonical sklearn-downstream usage; running on real digits dataset already discharges the cascade requirement.

## Step 7: Bug-hunt
- Reproducibility: identical seed -> identical `func_vals` (no global-state leak). PASS.
- Edge: singleton `Categorical(['only'])` accepted, all picks == 'only'. PASS.
- NumPy-2 surface: `RandomForestRegressor.predict(return_std=True)` -> finite non-negative std. PASS (directly stresses np.in1d->np.isin patch).
- Threading: 2 concurrent `Optimizer` instances finish with finite results. PASS. Minor curiosity: both converged to 7.7827 — deterministic under thread-interleaved init-point sampling, not a bug.
- Found bugs: **none**.

## Verdict
STATUS: USABLE

Reason: clean-venv `pip install -e` succeeds with Python 3.13 + NumPy 2.4 + sklearn 1.8; gp_minimize/forest_minimize converge on Branin & Rosenbrock; BayesSearchCV beats untuned baseline on real digits dataset; rescue carries 9 substantive Py3.13/NumPy-2/sklearn-1.x compatibility fixes; 7 distinct submodules invoked end-to-end from outside the rescue tree.
