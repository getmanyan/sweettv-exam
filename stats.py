"""Behavioural statistics for the report (validation split): how familiar the true and recommended lists are,
catalogue coverage, and MAP@5 by history size. Usage: python stats.py -> results/stats.md"""
import json
import os

import numpy as np
import pandas as pd

from common import K, apk, fallback_ranking
from evaluate import RESULTS, setup
from hybrid import maxnorm, method_scores, topk

ORDER = ["popularity", "history", "ppr", "als", "cbf", "assoc_rules", "fusion"]


def main():
    sch, ua, split, labels, users, test_users = setup()
    rows_all = split.users.get_indexer(users)
    warm = [u for u, r in zip(users, rows_all) if r >= 0]
    rows = split.users.get_indexer(warm)
    fb = split.items[fallback_ranking(split)].tolist()
    S = method_scores(split, rows)
    weights = json.load(open(os.path.join(RESULTS, "fusion.json")))["weights"]
    preds = {m: dict(zip(warm, topk(split, S[m], fb))) for m in S}
    preds["fusion"] = dict(zip(warm, topk(split, sum(w * maxnorm(S[m]) for m, w in weights.items() if w), fb)))
    seen = split.hist.groupby("user_id").tv_show_id.agg(set)
    n_shows = seen.map(len)
    out = ["# Validation statistics (warm users: %d of %d evaluated)" % (len(warm), len(users)), ""]
    lab_rep = np.mean([np.mean([i in seen[u] for i in labels[u]]) for u in warm])
    out.append(f"Share of true top-5 programmes (weeks 19-30) that the user already watched in weeks 11-18: {lab_rep:.3f}")
    out.append("")
    groups = {"1-4 programmes": (1, 4), "5-14": (5, 14), "15-29": (15, 29), "30+": (30, 10 ** 6)}
    gu = {g: [u for u in warm if lo <= n_shows[u] <= hi] for g, (lo, hi) in groups.items()}
    out += ["| Method | repeat share of slots | distinct programmes recommended | MAP@5 warm | "
            + " | ".join(f"MAP@5 {g} (n={len(v)})" for g, v in gu.items()) + " |",
            "|---" * (4 + len(gu)) + "|"]
    for m in ORDER:
        p = preds[m]
        rep = np.mean([np.mean([i in seen[u] for i in p[u]]) for u in warm])
        cov = len({i for u in warm for i in p[u]})
        allm = np.mean([apk(labels[u], p[u]) for u in warm])
        gm = [np.mean([apk(labels[u], p[u]) for u in v]) for v in gu.values()]
        out.append(f"| {m} | {rep:.3f} | {cov} | {allm:.4f} | " + " | ".join(f"{x:.4f}" for x in gm) + " |")
    test_new = set(sch[sch.week >= 31].tv_show_id) - set(sch[sch.week <= 30].tv_show_id)
    out += ["", f"Programmes airing in weeks 31-42 that never aired in weeks 11-30: {len(test_new)} "
            f"of {sch[sch.week >= 31].tv_show_id.nunique()}"]
    open(os.path.join(RESULTS, "stats.md"), "w").write("\n".join(out))
    print("\n".join(out))


if __name__ == "__main__":
    main()
