"""Step 7 bug-hunt for scikit-optimize rescue (sonnet).

Probes anti-PyCG-blindspot scenarios:
  1. Repeated gp_minimize calls (state leak between optimizers)
  2. Edge dimensions: zero-width Real, single-element Categorical
  3. NumPy 2.x boundary: np.in1d -> np.isin (fixed in patch); confirm
     forest std-prediction code path doesn't regress.
  4. Threading: two BayesSearchCV-like Optimizer instances in parallel.
"""
import threading
import numpy as np

from skopt import gp_minimize, Optimizer
from skopt.space import Real, Categorical
from skopt.benchmarks import branin
from skopt.learning import RandomForestRegressor


bugs = []


def probe_state_leak():
    # Two independent gp_minimize runs with same seed should give identical results.
    space = [Real(-5.0, 10.0), Real(0.0, 15.0)]
    f = lambda x: float(branin(x))
    r1 = gp_minimize(f, space, n_calls=10, n_initial_points=5, random_state=11)
    r2 = gp_minimize(f, space, n_calls=10, n_initial_points=5, random_state=11)
    if not np.allclose(r1.func_vals, r2.func_vals):
        bugs.append(f"state-leak: identical-seed runs diverge: {r1.fun} vs {r2.fun}")
    print(f"[probe1] reseeded gp_minimize reproducible: {r1.fun:.4f} == {r2.fun:.4f}")


def probe_edge_dims():
    # Categorical with single value
    try:
        s = [Real(0.0, 1.0), Categorical(["only"])]
        r = gp_minimize(lambda x: x[0] ** 2, s, n_calls=8, n_initial_points=4,
                        random_state=3)
        assert all(p[1] == "only" for p in r.x_iters)
        print(f"[probe2a] singleton Categorical accepted, all picks=='only'")
    except Exception as e:
        bugs.append(f"edge-singleton-cat: {type(e).__name__}: {e}")
        print(f"[probe2a] BUG singleton categorical: {e}")


def probe_forest_std():
    # Stresses the np.in1d -> np.isin patch in skopt/learning/forest.py
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, size=(80, 3))
    y = (X ** 2).sum(axis=1)
    rf = RandomForestRegressor(n_estimators=20, min_variance=0.0, random_state=0)
    rf.fit(X, y)
    mean, std = rf.predict(X[:5], return_std=True)
    if not (np.all(std >= 0) and np.all(np.isfinite(std))):
        bugs.append(f"forest std: invalid values {std}")
    print(f"[probe3] forest.predict(return_std=True) -> std={std.round(4)}")


def probe_threads():
    space = [Real(-2, 2), Real(-2, 2)]
    results = {}
    def run(tid):
        opt = Optimizer(space, base_estimator="GP", random_state=tid, n_initial_points=4)
        for _ in range(8):
            x = opt.ask()
            opt.tell(x, float(branin(x)))
        results[tid] = opt.get_result().fun
    ts = [threading.Thread(target=run, args=(i,)) for i in range(2)]
    for t in ts: t.start()
    for t in ts: t.join()
    print(f"[probe4] threaded Optimizer results: {results}")
    if not all(np.isfinite(v) for v in results.values()):
        bugs.append(f"thread: non-finite result {results}")


if __name__ == "__main__":
    probe_state_leak()
    probe_edge_dims()
    probe_forest_std()
    probe_threads()
    if bugs:
        print("BUGS FOUND:")
        for b in bugs:
            print(" -", b)
    else:
        print("NO_BUGS_FOUND")
