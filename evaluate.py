"""Validation harness. Temporal hold-out: history = weeks 11-18, target = weeks 19-30 (12 weeks, same horizon
as the Kaggle test window 31-42). Labels follow the competition recipe (see common.make_labels).
Every method gets the same users, the same candidate pool (shows airing in the target weeks, repeats allowed),
the same popularity fallback and the same MAP@5.

Usage: python evaluate.py [--grid]   (writes results/val_results.csv)
"""
import argparse
import csv
import os
import platform
import time

import numpy as np

import methods as M
from common import DATA, K, RESULTS, SEED, build_split, fallback_ranking, load_joined, make_labels, mapk, topk_from_scores

HIST_WEEKS = range(11, 19)
TARGET_WEEKS = range(19, 31)


def predict(split, scorer, users, batch: int = 1000):
    """Top-5 for each user; users without history (cold start) get the popularity fallback."""
    fb = fallback_ranking(split)
    rows = split.users.get_indexer(users)
    preds = {}
    warm = [(u, r) for u, r in zip(users, rows) if r >= 0]
    for s in range(0, len(warm), batch):
        chunk = warm[s:s + batch]
        rr = np.array([r for _, r in chunk])
        tops = topk_from_scores(split, scorer(rr), rr, fb)
        for (u, _), t in zip(chunk, tops):
            preds[u] = t
    cold_list = split.items[fb[:K]].tolist()
    for u, r in zip(users, rows):
        if r < 0:
            preds[u] = cold_list
    return preds


def setup():
    sch, ua = load_joined()
    split = build_split(sch, ua, HIST_WEEKS, TARGET_WEEKS, "val")
    labels = make_labels(ua, TARGET_WEEKS)
    users = sorted(labels)
    test_users = {u.split(",")[0] for u in open(os.path.join(DATA, "submission.csv")).read().splitlines()[1:]}
    return sch, ua, split, labels, users, test_users


def run(name, fitter, params, split, labels, users, test_users, writer=None):
    t0 = time.time()
    scorer = fitter(split, **params)
    t_fit = time.time() - t0
    preds = predict(split, scorer, users)
    t_all = time.time() - t0
    rows = split.users.get_indexer(users)
    warm = [u for u, r in zip(users, rows) if r >= 0]
    sub = [u for u in users if u in test_users]
    res = dict(method=name, params=str(params), map5=round(mapk(labels, preds), 5),
               map5_warm=round(mapk(labels, preds, users=warm), 5),
               map5_testusers=round(mapk(labels, preds, users=sub), 5) if sub else None,
               fit_s=round(t_fit, 1), total_s=round(t_all, 1))
    print(res, flush=True)
    if writer:
        writer.writerow(res)
    return preds, res


# Final comparison table: each method at the setting selected by tune.py on this same split
# (see results/val_grid.csv for every setting tried).
DEFAULTS = [
    ("popularity", M.fit_popularity, {}),
    ("popularity_x_airings", M.fit_popularity, {"by_airings": True}),
    ("history_count", M.fit_history, {"mode": "count"}),
    ("history_rate", M.fit_history, {"mode": "rate", "half_life": None, "smooth": 40.0}),
    ("ppr", M.fit_ppr, {"alpha": 0.5, "beta": 0.0, "log_weights": False}),
    ("als", M.fit_als, {"factors": 8, "alpha": 1.0, "reg": 30.0, "iters": 10}),
    ("cbf", M.fit_cbf, {"use_channel": True}),
    ("assoc_rules", M.fit_assoc, {"min_support": 10, "min_conf": 0.05, "agg": "max"}),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--out", default="val_results.csv")
    args = ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True)
    np.random.seed(SEED)
    t0 = time.time()
    sch, ua, split, labels, users, test_users = setup()
    rows = split.users.get_indexer(users)
    print(f"split {split.name}: hist weeks {split.hist_weeks[0]}-{split.hist_weeks[-1]}, "
          f"target {split.target_weeks[0]}-{split.target_weeks[-1]}; history users {len(split.users):,}, "
          f"items {len(split.items):,}, candidates {len(split.candidates):,}; eval users {len(users):,} "
          f"(cold {int((rows < 0).sum())}, also in Kaggle test {sum(u in test_users for u in users)}); "
          f"setup {time.time() - t0:.1f}s; python {platform.python_version()}")
    path = os.path.join(RESULTS, args.out)
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "params", "map5", "map5_warm", "map5_testusers", "fit_s", "total_s"])
        if new:
            w.writeheader()
        for name, fitter, params in DEFAULTS:
            if args.only and name not in args.only:
                continue
            run(name, fitter, params, split, labels, users, test_users, w)
            f.flush()


if __name__ == "__main__":
    main()
