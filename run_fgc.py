"""
Runnable driver for FGC (Featured Graph Coarsening), from
"Featured Graph Coarsening with Similarity Guarantees" (Kumar et al., ICML 2023)
/ "A Unified Framework for Optimization-Based Graph Coarsening" (JMLR).

The original repo ships only Jupyter notebooks whose cells are mutually
exclusive (several datasets each overwrite L and X) and which contain IPython
magics, so they cannot be executed linearly. This script:
  1. loads one dataset,
  2. builds the Laplacian L and feature matrix X,
  3. runs the UNMODIFIED solver_v2 extracted into fgc_core.py,
  4. reports the paper's metrics (relative eigen error, Dirichlet energy).

Usage:
  python run_fgc.py --dataset cora --ratio 0.3 --iterations 10
"""
import argparse
import time

import numpy as np

# --- Compatibility shim for NumPy >= 1.24 ---
# deeprobust (and other older deps) still use the removed scalar aliases
# np.int / np.float / np.bool. Restore them before those modules are imported.
for _alias, _target in [("int", int), ("float", float), ("bool", bool),
                        ("object", object), ("complex", complex), ("long", int)]:
    if not hasattr(np, _alias):
        setattr(np, _alias, _target)

import fgc_core


def load_dataset(name):
    """Return (L, X, n_nodes) for the requested dataset."""
    if name in ("cora", "citeseer", "polblogs"):
        from deeprobust.graph.data import Dataset
        data = Dataset(root="", name=name, setting="gcn", seed=10)
        A = np.array(data.adj.todense(), dtype=float)
        n = A.shape[0]
        # Laplacian L = D - A   (exactly as the notebook builds it)
        z = A @ np.ones(n)
        L = np.diag(z) - A

        if name == "polblogs":
            # polblogs has no node features: the notebook generates them from
            # the Laplacian pseudo-inverse.
            n_feat = 5000
            X = np.random.multivariate_normal(np.zeros(n), np.linalg.pinv(L), n_feat).T
        else:
            X = np.array(data.features.todense(), dtype=float)
        return L, X, n

    elif name in ("airfoil", "minnesota", "bunny"):
        # These come from PyGSP (as in the notebook).
        from pygsp import graphs
        G = {"airfoil": graphs.Airfoil,
             "minnesota": graphs.Minnesota,
             "bunny": graphs.Bunny}[name]()
        L = G.L.toarray().astype(float)
        n = G.N
        # No native features -> generate smooth features from pinv(L).
        n_feat = 500
        X = np.random.multivariate_normal(np.zeros(n), np.linalg.pinv(L), n_feat).T
        return L, X, n

    else:
        raise NotImplementedError(f"Unsupported dataset: {name}")


def eigen_error(L, C, top=100):
    """Relative eigenvalue error between L and the coarsened C^T L C,
    over the top-`top` eigenvalues (the notebook's REE computation)."""
    ev_orig = np.sort(np.linalg.eigvals(L).real)
    ev_coar = np.sort(np.linalg.eigvals(C.T @ L @ C).real)
    m = min(top, len(ev_coar))
    s, z = ev_orig[-m:], ev_coar[-m:]
    return float(np.mean(np.abs(z - s) / np.abs(s)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cora",
                    choices=["cora", "citeseer", "polblogs",
                             "airfoil", "minnesota", "bunny"])
    ap.add_argument("--ratio", type=float, default=0.3,
                    help="coarsening ratio r; coarse graph has k = r*n nodes")
    ap.add_argument("--iterations", type=int, default=10,
                    help="outer iterations of the FGC objective")
    ap.add_argument("--lambda_param", type=float, default=500.0)
    ap.add_argument("--alpha_param", type=float, default=500.0)
    ap.add_argument("--gamma_param", type=float, default=None,
                    help="default = X.shape[1]/2 (as in the notebook)")
    ap.add_argument("--seed", type=int, default=10)
    ap.add_argument("--eig", action="store_true",
                    help="compute eigen-error (dense eig; slow on big graphs)")
    ap.add_argument("--inner_iters", type=int, default=None,
                    help="inner update_C steps per outer iteration. The paper's "
                         "notebook hardcodes 100 (the default when omitted). "
                         "Lower values are much faster and useful for smoke tests.")
    args = ap.parse_args()

    np.random.seed(args.seed)

    print("=" * 62)
    print(f"FGC  |  dataset={args.dataset}  ratio={args.ratio}")
    print("=" * 62)

    t0 = time.time()
    L, X, n = load_dataset(args.dataset)
    print(f"loaded: n_nodes={n}, feature_dim={X.shape[1]}, L={L.shape}")

    k = int(round(args.ratio * n))
    gamma = args.gamma_param if args.gamma_param is not None else X.shape[1] / 2

    # The extracted solver reads module-level globals L and k (as in the notebook).
    fgc_core.L = L
    fgc_core.k = k

    print(f"coarsening: {n} -> {k} supernodes "
          f"(lambda={args.lambda_param}, alpha={args.alpha_param}, gamma={gamma})")

    obj = fgc_core.solver_v2(X, k, args.lambda_param, 0, args.alpha_param, gamma)

    t1 = time.time()
    if args.inner_iters is None:
        # Verbatim notebook behaviour (100 inner update_C steps per iteration).
        C, X_tilde, loss_ls = obj.fit(args.iterations)
    else:
        # Same update sequence as solver_v2.fit, with a configurable inner count.
        from tqdm import tqdm
        loss_ls = []
        for _ in tqdm(range(args.iterations)):
            for _ in range(args.inner_iters):
                obj.update_C(1 / obj.k)
            obj.update_X_tilde()
            loss_ls.append(obj.calc_f())
            obj.iters += 1
        C, X_tilde = obj.C, obj.X_tilde
    t2 = time.time()

    print("-" * 62)
    print(f"C shape (p x k):        {C.shape}")
    print(f"X_tilde shape (k x n):  {X_tilde.shape}")
    print(f"loss: first={loss_ls[0]:.4f}  last={loss_ls[-1]:.4f}  "
          f"(reduced by {loss_ls[0] - loss_ls[-1]:.4f})")
    print(f"time: load={t1-t0:.1f}s  optimize={t2-t1:.1f}s")

    # Dirichlet-energy style check from the notebook
    tr_coarse = np.trace(X_tilde.T @ C.T @ L @ C @ X_tilde)
    tr_orig = np.trace(X.T @ L @ X)
    print(f"tr(X~' C' L C X~) = {tr_coarse:.4f}")
    print(f"tr(X' L X)        = {tr_orig:.4f}")

    if args.eig:
        print("computing eigen-error (dense eigendecomposition)...")
        ee = eigen_error(L, C, top=100)
        print(f"relative eigen error (top-100): {ee:.6f}")

    # Supernode assignment: argmax over columns of C gives each node's supernode
    assign = np.asarray(C).argmax(axis=1)
    used = len(np.unique(assign))
    print(f"non-empty supernodes: {used} / {k}")
    print("=" * 62)
    print("FGC RUN COMPLETE")


if __name__ == "__main__":
    main()
