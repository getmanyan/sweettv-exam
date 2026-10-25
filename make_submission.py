"""Refit on all training weeks (11-30) and predict weeks 31-42 for the 1,260 Kaggle users.

Usage:
  python make_submission.py fusion            # final hybrid with the weights in results/fusion.json
  python make_submission.py single history    # any single method from hybrid.BEST
Writes submissions/<name>.csv and checks: header, 1,260 users in template order, 5 distinct ids per row,
every id airs in weeks 31-42.
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

from common import DATA, K, RESULTS, ROOT, build_split, fallback_ranking, load_joined
from hybrid import BEST, maxnorm, method_scores, topk

HIST_WEEKS = range(11, 31)
TARGET_WEEKS = range(31, 43)
OUT = str(ROOT / "submissions")
VAL_HIST_WEEKS = 8  # length of the validation history; 'smooth' was tuned for this length


def scale_params(split):
    """'smooth' in history_rate is in units of airings; the full history is 20 weeks instead of 8,
    so the same per-week prior needs a proportionally larger value."""
    fitter, params = BEST["history"]
    if params.get("mode") == "rate":
        params = dict(params, smooth=params["smooth"] * len(split.hist_weeks) / VAL_HIST_WEEKS)
    BEST["history"] = (fitter, params)


def check(path, template_users, valid_items):
    sub = pd.read_csv(path, dtype=str)
    assert list(sub.columns) == ["user_id", "tv_show_id"], sub.columns
    assert sub.user_id.tolist() == template_users, "users differ from template"
    lists = sub.tv_show_id.str.split(" ")
    assert all(len(l) == K and len(set(l)) == K for l in lists), "need 5 distinct ids per row"
    ids = {int(x) for l in lists for x in l}
    assert ids <= valid_items, "an id does not air in weeks 31-42"
    assert 0 not in ids
    print(f"check ok: {len(sub)} rows, {len(ids)} distinct programmes recommended")


def main():
    mode = sys.argv[1]
    t0 = time.time()
    sch, ua = load_joined()
    split = build_split(sch, ua, HIST_WEEKS, TARGET_WEEKS, "final")
    scale_params(split)
    template = pd.read_csv(os.path.join(DATA, "submission.csv"), dtype=str)
    users = template.user_id.tolist()
    rows_all = split.users.get_indexer(users)
    warm = [u for u, r in zip(users, rows_all) if r >= 0]
    rows = split.users.get_indexer(warm)
    fb = split.items[fallback_ranking(split)].tolist()
    if mode == "fusion":
        weights = json.load(open(os.path.join(RESULTS, "fusion.json")))["weights"]
        names = [m for m, w in weights.items() if w]
        S = method_scores(split, rows, names)
        tot = sum(weights[m] * maxnorm(S[m]) for m in names)
        name = "fusion"
    else:
        m = sys.argv[2]
        S = method_scores(split, rows, [m])
        tot, name = S[m], m
    preds = dict(zip(warm, topk(split, tot, fb)))
    n_cold = 0
    for u, r in zip(users, rows_all):
        if r < 0:
            preds[u] = fb[:K]
            n_cold += 1
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"{name}.csv")
    pd.DataFrame({"user_id": users, "tv_show_id": [" ".join(map(str, preds[u])) for u in users]}).to_csv(path, index=False)
    valid = set(split.items[split.candidates].tolist())
    check(path, users, valid)
    print(f"{path}: cold-start users {n_cold}, params {BEST if mode != 'fusion' else weights}, {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
