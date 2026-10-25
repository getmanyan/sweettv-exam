"""Hybrid strategies on top of the individual methods.

  fusion   : score fusion. Each method's scores are max-normalised per user over the candidate pool,
             then combined with weights. Weights are chosen on half A of the validation users and
             reported on half B (and on all users) so the choice is not judged on the users it was tuned on.
  reranker : candidate generation (top-N of every method) + LightGBM LambdaRank on per-pair features.
             Trained on half A users (weeks 11-18 features -> weeks 19-30 labels), evaluated on half B.

Usage: python hybrid.py fusion|reranker
"""
import itertools
import json
import os
import sys
import time

import numpy as np
import pandas as pd

import methods as M
from common import K, SEED, apk, fallback_ranking, mapk
from evaluate import RESULTS, setup

# best single-method settings from results/val_grid.csv (filled in after the grid search)
BEST = {
    "popularity": (M.fit_popularity, {"by_airings": True}),
    "history": (M.fit_history, {"mode": "rate", "half_life": None, "smooth": 40.0}),
    "ppr": (M.fit_ppr, {"alpha": 0.5, "beta": 0.0, "log_weights": False}),
    "als": (M.fit_als, {"factors": 8, "alpha": 1.0, "reg": 30.0, "iters": 10}),
    "cbf": (M.fit_cbf, {"use_channel": True}),
    "assoc_rules": (M.fit_assoc, {"min_support": 10, "min_conf": 0.05, "agg": "max"}),
}


def method_scores(split, rows, names=None, batch=1000):
    """Dense scores restricted to the candidate pool: {name: (len(rows), n_candidates) float32}."""
    out = {}
    for name in names or BEST:
        fitter, params = BEST[name]
        t0 = time.time()
        sc = fitter(split, **params)
        parts = [sc(rows[s:s + batch])[:, split.candidates].astype(np.float32) for s in range(0, len(rows), batch)]
        out[name] = np.vstack(parts)
        print(f"  scored {name} in {time.time() - t0:.1f}s", flush=True)
    return out


def maxnorm(S):
    m = S.max(axis=1, keepdims=True)
    m[m <= 0] = 1.0
    return np.clip(S, 0, None) / m


def topk(split, S, fb_items, k=K):
    """S: (n_rows, n_candidates). Positive scores first, popularity fallback fills the rest."""
    idx = np.argsort(-S, axis=1, kind="stable")[:, : k * 3]
    cand_items = split.items[split.candidates].to_numpy()
    res = []
    for r in range(S.shape[0]):
        pick = [cand_items[c] for c in idx[r] if S[r, c] > 0][:k]
        for it in fb_items:
            if len(pick) == k:
                break
            if it not in pick:
                pick.append(it)
        res.append(pick)
    return res


def halves(users):
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(users))
    a = sorted(users[i] for i in perm[: len(users) // 2])
    b = sorted(users[i] for i in perm[len(users) // 2:])
    return a, b


def run_fusion():
    sch, ua, split, labels, users, test_users = setup()
    rows_all = split.users.get_indexer(users)
    warm = [u for u, r in zip(users, rows_all) if r >= 0]
    rows = split.users.get_indexer(warm)
    fb = split.items[fallback_ranking(split)].tolist()
    cold = {u: fb[:K] for u, r in zip(users, rows_all) if r < 0}
    S = {k: maxnorm(v) for k, v in method_scores(split, rows).items()}
    A, B = halves(users)
    pos = {u: i for i, u in enumerate(warm)}

    def evaluate(weights):
        tot = sum(w * S[m] for m, w in weights.items() if w)
        preds = dict(zip(warm, topk(split, tot, fb)))
        preds.update(cold)
        return preds

    results = []
    grid = {"ppr": [0, 0.5, 1, 2], "assoc_rules": [0, 0.25, 0.5, 1], "cbf": [0, 0.1, 0.25, 0.5],
            "als": [0, 0.25], "popularity": [0, 0.1]}
    t0 = time.time()
    for vals in itertools.product(*grid.values()):
        w = dict(zip(grid, vals))
        w["history"] = 1.0
        p = evaluate(w)
        results.append((mapk(labels, p, users=A), w, p))
    results.sort(key=lambda x: -x[0])
    best_a, best_w, best_p = results[0]
    report = {
        "weights": best_w,
        "map5_half_A_tuned": round(best_a, 5),
        "map5_half_B_heldout": round(mapk(labels, best_p, users=B), 5),
        "map5_all": round(mapk(labels, best_p), 5),
        "map5_testusers": round(mapk(labels, best_p, users=[u for u in users if u in test_users]), 5),
        "grid_size": len(results), "grid_seconds": round(time.time() - t0, 1),
    }
    # each method alone through the same normalisation, on the same halves, for the comparison table
    for m in S:
        p = evaluate({m: 1.0})
        report[f"single_{m}"] = {"A": round(mapk(labels, p, users=A), 5), "B": round(mapk(labels, p, users=B), 5)}
    print(json.dumps(report, indent=1, ensure_ascii=False))
    print("top-5 weight sets on half A:")
    for s, w, p in results[:5]:
        print(round(s, 5), round(mapk(labels, p, users=B), 5), w)
    with open(os.path.join(RESULTS, "fusion.json"), "w") as f:
        json.dump(report, f, indent=1, ensure_ascii=False)


def run_switch():
    """Switching hybrid: users with fewer than T distinct programmes in history get the list of a
    'thin-history' method (association rules or popularity x airings); everyone else gets the fusion.
    T and the thin-history method are chosen on half A users and reported on half B."""
    sch, ua, split, labels, users, test_users = setup()
    rows_all = split.users.get_indexer(users)
    warm = [u for u, r in zip(users, rows_all) if r >= 0]
    rows = split.users.get_indexer(warm)
    fb = split.items[fallback_ranking(split)].tolist()
    cold = {u: fb[:K] for u, r in zip(users, rows_all) if r < 0}
    S = {k: maxnorm(v) for k, v in method_scores(split, rows).items()}
    weights = json.load(open(os.path.join(RESULTS, "fusion.json")))["weights"]
    base = dict(zip(warm, topk(split, sum(w * S[m] for m, w in weights.items() if w), fb)))
    alts = {m: dict(zip(warm, topk(split, S[m], fb))) for m in ["assoc_rules", "popularity", "ppr"]}
    n_shows = split.hist.groupby("user_id").tv_show_id.nunique()
    A, B = halves(users)
    res = []
    for m, alt in alts.items():
        for T in [0, 2, 3, 4, 5, 6, 8, 10, 15]:
            p = {u: (alt[u] if n_shows.get(u, 0) < T else base[u]) for u in warm}
            p.update(cold)
            res.append((mapk(labels, p, users=A), mapk(labels, p, users=B), mapk(labels, p), m, T,
                        sum(n_shows.get(u, 0) < T for u in warm)))
    res.sort(key=lambda x: -x[0])
    print("A(tuned)  B(held-out)  all  thin-method  T  switched-users")
    for r in res[:10]:
        print(f"{r[0]:.5f}  {r[1]:.5f}  {r[2]:.5f}  {r[3]}  {r[4]}  {r[5]}")
    base_row = [r for r in res if r[4] == 0][0]
    print(f"no switching: A {base_row[0]:.5f} B {base_row[1]:.5f} all {base_row[2]:.5f}")
    best = res[0]
    out = {"thin_method": best[3], "T": int(best[4]), "map5_half_A_tuned": round(best[0], 5),
           "map5_half_B_heldout": round(best[1], 5), "map5_all": round(best[2], 5), "switched_users": int(best[5]),
           "no_switch": {"A": round(base_row[0], 5), "B": round(base_row[1], 5), "all": round(base_row[2], 5)}}
    json.dump(out, open(os.path.join(RESULTS, "switch.json"), "w"), indent=1)


if __name__ == "__main__":
    {"fusion": run_fusion, "switch": run_switch}[sys.argv[1]]()
