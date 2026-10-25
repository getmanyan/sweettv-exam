"""Shared pieces: temporal split, label construction (competition recipe), MAP@5, top-k helpers."""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import scipy.sparse as sp

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# override with env vars if the data lives elsewhere
DATA = os.environ.get("SWEETTV_DATA", str(ROOT / "data"))
WORK = os.environ.get("SWEETTV_WORK", str(ROOT / "work"))
RESULTS = str(ROOT / "results")
K = 5
SEED = 42


def load_joined(work: str = WORK):
    sch = pd.read_pickle(f"{work}/airings.pkl")
    ua = pd.read_pickle(f"{work}/user_airings.pkl")
    return sch, ua


def make_labels(ua: pd.DataFrame, weeks, k: int = K) -> dict:
    """Competition recipe: count airings watched >= threshold per (user, show) inside `weeks`,
    keep users with at least k distinct shows, take the k most frequent (ties: more seconds first)."""
    w = ua[ua.watched & ua.week.isin(weeks)]
    g = w.groupby(["user_id", "tv_show_id"]).agg(n=("airing_id", "size"), sec=("overlap", "sum")).reset_index()
    g = g.sort_values(["user_id", "n", "sec"], ascending=[True, False, False])
    labels = {}
    for u, grp in g.groupby("user_id", sort=False):
        if len(grp) >= k:
            labels[u] = grp.tv_show_id.to_numpy()[:k].tolist()
    return labels


def apk(actual, predicted, k: int = K) -> float:
    """Average precision at k, same definition as the Kaggle MAP@K metric."""
    predicted = list(predicted)[:k]
    score, hits = 0.0, 0
    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            hits += 1
            score += hits / (i + 1.0)
    return score / min(len(actual), k) if actual else 0.0


def mapk(labels: dict, preds: dict, k: int = K, users=None) -> float:
    users = list(labels) if users is None else list(users)
    return float(np.mean([apk(labels[u], preds.get(u, []), k) for u in users]))


@dataclass
class Split:
    """Everything a method may see: history interactions plus the target-window schedule (what airs, not who watched)."""
    name: str
    hist_weeks: list
    target_weeks: list
    hist: pd.DataFrame            # watched user-airing rows inside hist_weeks
    users: pd.Index               # user index (all users with any history row)
    items: pd.Index               # item index = shows that air in hist or target weeks
    X: sp.csr_matrix              # users x items, count of watched airings in history
    candidates: np.ndarray        # item positions airing in the target window (the candidate pool)
    air_hist: np.ndarray          # number of airings per item in hist weeks
    air_target: np.ndarray        # number of airings per item in target weeks
    item_meta: pd.DataFrame       # one row of metadata per item (title, category, genres, ...)
    hist_week_of: dict = field(default_factory=dict)


def build_split(sch: pd.DataFrame, ua: pd.DataFrame, hist_weeks, target_weeks, name: str) -> Split:
    hist_weeks, target_weeks = list(hist_weeks), list(target_weeks)
    hist = ua[ua.watched & ua.week.isin(hist_weeks)].copy()
    sch_h = sch[sch.week.isin(hist_weeks)]
    sch_t = sch[sch.week.isin(target_weeks)]
    items = pd.Index(pd.unique(np.concatenate([sch_h.tv_show_id.to_numpy(), sch_t.tv_show_id.to_numpy()])))
    users = pd.Index(sorted(hist.user_id.unique()))
    ui = users.get_indexer(hist.user_id)
    ii = items.get_indexer(hist.tv_show_id)
    X = sp.csr_matrix((np.ones(len(hist)), (ui, ii)), shape=(len(users), len(items)))
    X.sum_duplicates()
    air_hist = np.bincount(items.get_indexer(sch_h.tv_show_id), minlength=len(items)).astype(float)
    air_target = np.bincount(items.get_indexer(sch_t.tv_show_id), minlength=len(items)).astype(float)
    candidates = np.flatnonzero(air_target > 0)
    meta_cols = ["tv_show_id", "tv_show_title", "tv_show_category", "tv_show_genre_1", "tv_show_genre_2",
                 "tv_show_genre_3", "year_of_production", "director", "actors", "channel_title"]
    # metadata may only come from the schedule (hist + target airings), never from target viewing
    meta = pd.concat([sch_h, sch_t])[meta_cols].drop_duplicates("tv_show_id", keep="last")
    meta = meta.set_index("tv_show_id").reindex(items)
    return Split(name, hist_weeks, target_weeks, hist, users, items, X.tocsr(), candidates,
                 air_hist, air_target, meta)


def popularity_scores(split: Split) -> np.ndarray:
    """Reference ranking: number of distinct users who watched the show in history."""
    return np.asarray((split.X > 0).sum(axis=0)).ravel().astype(float)


def topk_from_scores(split: Split, scores: np.ndarray, user_rows: np.ndarray, fallback: np.ndarray, k: int = K):
    """scores: (len(user_rows), n_items). Restricts to the candidate pool, keeps positive scores,
    and fills any empty slots from the fallback ranking (popularity), without duplicates.

    Ties are broken deterministically by item position (lower index first). An earlier version pre-selected
    candidates with np.argpartition, which does not define an order among equal scores; on a Mac (ARM) it
    picked different tied programmes than on Linux (x86), so methods with many ties (history count, CBF,
    association rules) gave slightly different MAP@5 on the two machines. Found during the verification run."""
    out = []
    mask = np.full(scores.shape[1], -np.inf)
    mask[split.candidates] = 0.0
    # Round to 9 digits relative to each user's largest score: scores that are equal in exact arithmetic
    # (e.g. CBF cosines of programmes with identical metadata) can differ in the 16th digit between CPUs,
    # which a stable sort cannot see. Ranking is invariant to the per-row scaling.
    m = np.abs(scores).max(axis=1, keepdims=True)
    m[m == 0] = 1.0
    s = np.round(scores / m, 9) + mask
    order = np.argsort(-s, axis=1, kind="stable")[:, : k * 4]
    for r in range(s.shape[0]):
        cand = order[r]
        picked = [c for c in cand if s[r, c] > 0][:k]
        if len(picked) < k:
            for c in fallback:
                if c not in picked:
                    picked.append(c)
                if len(picked) == k:
                    break
        out.append(split.items[picked].tolist())
    return out


def fallback_ranking(split: Split) -> np.ndarray:
    pop = popularity_scores(split)
    cand = split.candidates
    return cand[np.argsort(-pop[cand], kind="stable")][:50]
