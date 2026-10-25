"""Two example users on the validation split (history weeks 11-18 -> actual weeks 19-30), every method side by side.

User selection rule (deterministic, documented in the report):
  candidates = users who are Kaggle test users AND have validation labels AND some history;
  heavy viewer = the candidate at the 90th percentile of watched airings in history,
  light viewer = the candidate with the fewest distinct programmes in history (ties: fewest airings, then user_id).
For each user: the history profile, the actual top-5 in weeks 19-30, each method's top-5 (id, title, category,
hit mark) and its AP@5. Also writes a hand-check sheet for one user so AP@5 can be recomputed by hand.

Usage: python examples.py   -> results/examples.md, results/handcheck.md
"""
import os

import numpy as np
import pandas as pd

from common import K, apk, fallback_ranking
from evaluate import RESULTS, setup
from hybrid import BEST, maxnorm, method_scores, topk
import json

ORDER = ["popularity", "history", "ppr", "als", "cbf", "assoc_rules", "fusion"]


def pick_users(split, labels, test_users):
    h = split.hist
    stats = h.groupby("user_id").agg(airings=("airing_id", "size"), shows=("tv_show_id", "nunique"))
    cand = stats[stats.index.isin(test_users) & stats.index.isin(list(labels))]
    q = cand.airings.quantile(0.9)
    heavy = (cand.airings - q).abs().sort_values(kind="stable").index[0]
    light = cand.sort_values(["shows", "airings"], kind="stable").reset_index().iloc[0].user_id
    return heavy, light, cand


def show(split, item_id):
    m = split.item_meta.loc[item_id]
    title = str(m.tv_show_title)[:48]
    return title, m.tv_show_category


def main():
    sch, ua, split, labels, users, test_users = setup()
    heavy, light, cand = pick_users(split, labels, test_users)
    chosen = [("Heavy viewer", heavy), ("Light viewer", light)]
    rows = split.users.get_indexer([u for _, u in chosen])
    S = method_scores(split, rows)
    fb = split.items[fallback_ranking(split)].tolist()
    weights = json.load(open(os.path.join(RESULTS, "fusion.json")))["weights"]
    preds = {m: topk(split, S[m], fb) for m in S}
    preds["fusion"] = topk(split, sum(w * maxnorm(S[m]) for m, w in weights.items() if w), fb)

    lines = ["# Two example users (validation: history weeks 11-18, actual weeks 19-30)", "",
             f"Selection: {len(cand)} Kaggle test users with history and validation labels; heavy = 90th percentile "
             f"of watched airings in history ({cand.airings.quantile(0.9):.0f}), light = fewest distinct programmes.", ""]
    for r, (label, u) in enumerate(chosen):
        hu = split.hist[split.hist.user_id == u]
        top_hist = hu.groupby("tv_show_id").size().sort_values(ascending=False, kind="stable")
        lines += [f"## {label}: user {u}", "",
                  f"History weeks 11-18: {len(hu)} watched airings of {hu.tv_show_id.nunique()} programmes.", "",
                  "| History rank | id | title | category | watched airings |", "|---|---|---|---|---|"]
        for i, (it, n) in enumerate(top_hist.head(8).items(), 1):
            t, c = show(split, it)
            lines.append(f"| {i} | {it} | {t} | {c} | {n} |")
        actual = labels[u]
        lines += ["", "Actual top-5 in weeks 19-30 (competition recipe):", "",
                  "| # | id | title | category | in history? |", "|---|---|---|---|---|"]
        for i, it in enumerate(actual, 1):
            t, c = show(split, it)
            lines.append(f"| {i} | {it} | {t} | {c} | {'yes' if it in top_hist.index else 'no'} |")
        lines += ["", "| Method | #1 | #2 | #3 | #4 | #5 | AP@5 |", "|---|---|---|---|---|---|---|"]
        for m in ORDER:
            p = preds[m][r]
            cells = []
            for it in p:
                t, c = show(split, it)
                mark = "**HIT** " if it in actual else ""
                new = "" if it in top_hist.index else " (new)"
                cells.append(f"{mark}{it} {t} [{c}]{new}")
            lines.append(f"| {m} | " + " | ".join(cells) + f" | {apk(actual, p):.3f} |")
        lines.append("")
    open(os.path.join(RESULTS, "examples.md"), "w").write("\n".join(lines))

    # hand-check sheet: fusion on the heavy viewer
    u, p, actual = heavy, preds["fusion"][0], labels[heavy]
    hc = ["# Hand-check of AP@5 (fusion, heavy viewer)", "", f"user {u}", f"relevant (actual top-5): {[int(x) for x in actual]}",
          f"recommended: {[int(x) for x in p]}", "", "| rank | id | relevant? | hits so far | precision at rank |", "|---|---|---|---|---|"]
    hits, terms = 0, []
    for i, it in enumerate(p, 1):
        rel = it in actual
        hits += rel
        hc.append(f"| {i} | {it} | {'yes' if rel else 'no'} | {hits} | {hits}/{i} = {hits / i:.3f} |")
        if rel:
            terms.append(f"{hits}/{i}")
    hc += ["", f"AP@5 = ({' + '.join(terms) or '0'}) / min(5, {len(actual)}) = {apk(actual, p):.4f}  (value from common.apk)"]
    open(os.path.join(RESULTS, "handcheck.md"), "w").write("\n".join(hc))
    print("\n".join(lines))
    print("\n".join(hc))


if __name__ == "__main__":
    main()
