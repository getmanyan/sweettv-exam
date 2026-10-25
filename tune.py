"""Hyperparameter grids for every method on the validation split (selection only, all models fit on history).
Usage: python tune.py METHOD [METHOD ...]   -> appends rows to results/val_grid.csv"""
import csv
import itertools
import os
import sys

import methods as M
from evaluate import RESULTS, run, setup

GRIDS = {
    "popularity": (M.fit_popularity, {"by_airings": [False, True]}),
    "history": (M.fit_history, {"mode": ["count", "rate"], "half_life": [None, 2, 4, 8], "smooth": [0.0, 1.0, 3.0, 10.0]}),
    "ppr": (M.fit_ppr, {"alpha": [0.3, 0.5, 0.7, 0.85], "beta": [0.0, 0.25, 0.5], "log_weights": [True, False]}),
    "als": (M.fit_als, {"factors": [32, 64, 128], "alpha": [5.0, 20.0, 40.0], "reg": [0.1, 1.0], "iters": [10]}),
    "cbf": (M.fit_cbf, {"use_channel": [False, True]}),
    # extensions past the edge of the first grid (best values sat on the boundary)
    "history_ext": (M.fit_history, {"mode": ["rate"], "half_life": [None], "smooth": [20.0, 40.0, 80.0]}),
    "als_ext": (M.fit_als, {"factors": [8, 16, 32], "alpha": [1.0, 2.0, 5.0], "reg": [1.0, 10.0], "iters": [10]}),
    "als_ext2": (M.fit_als, {"factors": [4, 8], "alpha": [0.5, 1.0], "reg": [10.0, 30.0, 100.0], "iters": [10]}),
    "assoc_rules_ext": (M.fit_assoc, {"min_support": [20, 40], "min_conf": [0.05], "agg": ["max"]}),
    "assoc_rules": (M.fit_assoc, {"min_support": [2, 5, 10], "min_conf": [0.01, 0.05, 0.1], "agg": ["sum", "max"]}),
}


def main():
    sch, ua, split, labels, users, test_users = setup()
    path = os.path.join(RESULTS, "val_grid.csv")
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "params", "map5", "map5_warm", "map5_testusers", "fit_s", "total_s"])
        if new:
            w.writeheader()
        for name in sys.argv[1:]:
            fitter, grid = GRIDS[name]
            keys = list(grid)
            for vals in itertools.product(*grid.values()):
                params = dict(zip(keys, vals))
                if name == "history" and params["mode"] == "count" and params["smooth"] != 0.0:
                    continue  # smooth only matters for mode='rate'
                run(name, fitter, params, split, labels, users, test_users, w)
                f.flush()


if __name__ == "__main__":
    main()
