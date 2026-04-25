# scikit-optimize (RepoRescue modernized fork)

Sequential model-based optimization (SMBO) for any scikit-learn-compatible objective: **Bayesian optimization** with Gaussian-process surrogates, **tree-of-Parzen** / random-forest surrogates, and a drop-in replacement for `GridSearchCV` (`BayesSearchCV`). Originally introduced at the NeurIPS 2017 BayesOpt workshop and the de-facto Python BO library since.

> **Modernized for NumPy 2.x / scikit-learn 1.8 / Python 3.13.** Upstream `scikit-optimize==0.9.0` has been unmaintained since 2021 and no longer imports against the current SciPy/sklearn stack. This fork patches the surfaces that broke without changing the public API.

---

## Install

```bash
python3.13 -m venv .venv && source .venv/bin/activate
pip install -e .
# pulls NumPy >= 2.4, SciPy >= 1.17, scikit-learn >= 1.8, joblib, pyaml
```

## Quick start: `gp_minimize` on Branin

```python
from skopt import gp_minimize, forest_minimize
from skopt.space import Real
from skopt.utils import use_named_args
from skopt.benchmarks import branin

space = [Real(-5.0, 10.0, name="x0"), Real(0.0, 15.0, name="x1")]

@use_named_args(space)
def f(x0, x1):
    return float(branin([x0, x1]))

res = gp_minimize(f, space, n_calls=20, n_initial_points=8,
                  acq_func="EI", random_state=42)
print(res.fun, res.x)  # ~0.886 in 20 calls (global min ~0.398)
assert res.fun < 2.0    # converges from f0 ~20.0
```

`forest_minimize` on 2-D Rosenbrock (`n_calls=30`) drops final `f` to **0.575**, beating best-of-10-random-init — the surrogate is doing useful work, not just sampling.

## What this fork fixes

Nine substantive Py3.13 / NumPy-2 / sklearn-1.x compatibility patches over the upstream `0.9.0` source tree:

| Surface | Site |
|---|---|
| `np.int` removed (NumPy 1.20+) -> Python `int` | `skopt/space/transformers.py:259,272` |
| `np.in1d` deprecated (NumPy 2.0) -> `np.isin` | `skopt/learning/forest.py:240` |
| sklearn 1.x: `criterion='mse'` -> `'squared_error'` | `skopt/learning/forest.py:127,189` (+6 sites) |
| sklearn 1.x: `max_features='auto'` -> `1.0` | `skopt/learning/forest.py:130,192` |
| sklearn 1.x estimator-tag system | `skopt/learning/gbrt.py:38-43` — `__sklearn_tags__` regressor method |
| sklearn 1.x: `is_regressor` strict typecheck | `skopt/utils.py:360-372`, `skopt/optimizer/optimizer.py:222-235` — duck-type guard |
| `distutils` removed (setuptools 58+ / Py 3.12) | `setup.py:43-46` — fallback path deleted |
| Python-2 `__builtin__` legacy | `setup.py:47-51` — collapsed to `import builtins` |
| `python_requires>=3.13` | `setup.py:83` |

Public API (`gp_minimize`, `forest_minimize`, `Optimizer`, `BayesSearchCV`, `Real/Integer/Categorical`) is unchanged.

## Validated downstream: `BayesSearchCV` on `load_digits`

Beyond unit tests, the modernized fork is exercised end-to-end as a sklearn hyperparameter-search drop-in (see `.reporescue/scenario_validate.py`): tuning `Pipeline(StandardScaler -> SVC)` over `{kernel, C, gamma, degree}` with `BayesSearchCV(n_iter=15, cv=3)` on `sklearn.datasets.load_digits`.

- Untuned 3-fold CV: `0.9733`
- BayesSearchCV best CV: `0.9755`
- **Held-out test accuracy: `0.9778`** (on 25% holdout, asserted > 0.90)
- Search explored `{linear, poly, rbf}` kernels, 15 distinct `C` values

## Bug-hunt probes

All four anti-regression probes pass (`.reporescue/bug_hunt.py`):

1. Reproducibility — identical `random_state` -> bit-identical `func_vals` (no global-state leak)
2. Edge dimensions — singleton `Categorical(['only'])` accepted, sampler honours degenerate domain
3. NumPy-2 surface — `RandomForestRegressor.predict(return_std=True)` returns finite, non-negative std (directly stresses the `np.in1d`->`np.isin` patch)
4. Threading — two concurrent `Optimizer` instances finish with finite results

## Disclaimer

This is a **community modernization fork** maintained under the [RepoRescue](https://github.com/RepoRescue) benchmark project to keep abandoned-but-still-useful Python tools alive on Python 3.13. It is **not** affiliated with or endorsed by the original scikit-optimize maintainers. Algorithmic behavior is preserved as-is from upstream `0.9.0`; the patches address only language/framework deprecations.

For new projects we still recommend evaluating actively-maintained alternatives (`optuna`, `ax-platform`, `botorch`). Use this fork when you need API compatibility with existing `skopt`-based code on a modern Python stack.

## License

BSD-3-Clause (inherited from upstream scikit-optimize). See `LICENSE`.
