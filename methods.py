"""Recommendation methods. Each `fit_*` takes a Split and returns score(rows) -> dense (len(rows), n_items).
All methods see only history interactions and the schedule (what airs when)."""
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from common import SEED, Split, popularity_scores


def _weighted_X(split: Split, half_life=None) -> sp.csr_matrix:
    """User x item matrix of watched-airing counts, optionally with exponential recency decay by week."""
    if half_life is None:
        return split.X
    h = split.hist
    last = max(split.hist_weeks)
    w = 0.5 ** ((last - h.week.to_numpy()) / half_life)
    ui = split.users.get_indexer(h.user_id)
    ii = split.items.get_indexer(h.tv_show_id)
    X = sp.csr_matrix((w, (ui, ii)), shape=split.X.shape)
    X.sum_duplicates()
    return X


# ---------------------------------------------------------------- reference and habit baselines

def fit_popularity(split: Split, by_airings: bool = False):
    pop = popularity_scores(split)
    if by_airings:  # popular per airing x how often it airs in the target window
        pop = pop / np.maximum(split.air_hist, 1) * split.air_target
    return lambda rows: np.tile(pop, (len(rows), 1))


def fit_history(split: Split, half_life=None, mode: str = "count", smooth: float = 0.0):
    """Personal history ('habit'): the user's own watched-airing counts.
    mode='count': raw (recency-weighted) count; mode='rate': count / airings in history x airings in target,
    i.e. the user's watch rate per airing extrapolated to how often the show airs in the target window."""
    X = _weighted_X(split, half_life)
    if mode == "rate":
        f = split.air_target / (split.air_hist + smooth + 1e-9)
        X = X @ sp.diags(f)
    X = X.tocsr()
    return lambda rows: X[rows].toarray()


# ---------------------------------------------------------------- Personalized PageRank

def fit_ppr(split: Split, alpha: float = 0.5, iters: int = 20, beta: float = 0.0, log_weights: bool = True):
    """Random walk with restart on the user-programme bipartite graph, observed on the programme side:
    one step = programme -> user who watched it -> another programme that user watched.
    Restart distribution = the user's own (normalised) history. beta > 0 divides by programme degree^beta."""
    R = split.X.copy().astype(float)
    if log_weights:
        R.data = np.log1p(R.data)
    P_ui = normalize(R, norm="l1", axis=1).tocsr()          # user -> item
    P_iu = normalize(R.T.tocsr(), norm="l1", axis=1).tocsr()  # item -> user
    deg = np.asarray((R > 0).sum(axis=0)).ravel()
    pen = np.power(np.maximum(deg, 1), beta) if beta else None

    def score(rows):
        r0 = P_ui[rows].toarray()
        p = r0.copy()
        for _ in range(iters):
            p = (1 - alpha) * r0 + alpha * np.asarray((p @ P_iu) @ P_ui)
        return p / pen if pen is not None else p
    return score


# ---------------------------------------------------------------- Matrix factorisation (implicit ALS)

def fit_als(split: Split, factors: int = 64, reg: float = 0.1, alpha: float = 10.0, iters: int = 15, seed: int = SEED):
    """Hu, Koren & Volinsky (2008) implicit-feedback ALS. Confidence c = 1 + alpha * log(1 + count)."""
    R = split.X.tocsr().astype(float)
    C = R.copy()
    C.data = alpha * np.log1p(C.data)          # c - 1
    Ct = C.T.tocsr()
    nu, ni = R.shape
    rng = np.random.default_rng(seed)
    U = rng.normal(0, 0.01, (nu, factors))
    V = rng.normal(0, 0.01, (ni, factors))
    I = reg * np.eye(factors)

    def solve(A, Cm, Y):
        YtY = Y.T @ Y
        out = np.zeros((A.shape[0], factors))
        for u in range(A.shape[0]):
            s, e = Cm.indptr[u], Cm.indptr[u + 1]
            idx, c1 = Cm.indices[s:e], Cm.data[s:e]
            if len(idx) == 0:
                continue
            Yu = Y[idx]
            M = YtY + (Yu.T * c1) @ Yu + I
            b = Yu.T @ (1 + c1)
            out[u] = np.linalg.solve(M, b)
        return out

    for _ in range(iters):
        U = solve(U, C, V)
        V = solve(V, Ct, U)
    return lambda rows: U[rows] @ V.T


# ---------------------------------------------------------------- Content-based filtering

def _tokens(row) -> str:
    toks = []
    if isinstance(row.tv_show_category, str):
        toks.append("cat_" + row.tv_show_category)
    for g in (row.tv_show_genre_1, row.tv_show_genre_2, row.tv_show_genre_3):
        if isinstance(g, str):
            toks.append("gen_" + g)
    for col, pre in (("director", "dir_"), ("actors", "act_")):
        v = getattr(row, col)
        if isinstance(v, str):
            toks += [pre + x.strip() for x in v.split(",") if x.strip()]
    y = str(row.year_of_production) if pd.notna(row.year_of_production) else ""
    if y[:4].isdigit():
        toks.append(f"dec_{y[:3]}0s")
    return " ".join(t.replace(" ", "_") for t in toks)


def fit_cbf(split: Split, use_channel: bool = False):
    """TF-IDF item profiles from schedule metadata (category, genres, director, actors, decade[, channel]).
    User profile = history-weighted sum of item vectors; score = cosine(user profile, item)."""
    docs = []
    for row in split.item_meta.itertuples():
        d = _tokens(row)
        if use_channel and isinstance(row.channel_title, str):
            d += " ch_" + row.channel_title.replace(" ", "_")
        docs.append(d if d else "none")
    vec = TfidfVectorizer(token_pattern=r"\S+", lowercase=False, sublinear_tf=True)
    F = vec.fit_transform(docs).tocsr()                      # rows L2-normalised
    W = split.X.copy().astype(float)
    W.data = np.log1p(W.data)
    prof = normalize(W @ F, norm="l2", axis=1).tocsr()
    FT = F.T.tocsc()

    def score(rows):
        return np.asarray((prof[rows] @ FT).todense())
    score.n_features = F.shape[1]
    return score


# ---------------------------------------------------------------- Association rules

def fit_assoc(split: Split, min_support: int = 5, min_conf: float = 0.05, agg: str = "sum"):
    """Pairwise rules a -> b mined from each user's set of watched programmes in history.
    support(a,b) = #users with both; confidence = support(a,b) / support(a). Rules below the thresholds are dropped.
    Score(u, b) = sum (or max) of confidence over antecedents a in the user's history."""
    B = (split.X > 0).astype(float).tocsr()
    sup = np.asarray(B.sum(axis=0)).ravel()
    C = (B.T @ B).tocsr()
    C.setdiag(0)
    C.eliminate_zeros()
    C.data[C.data < min_support] = 0
    C.eliminate_zeros()
    conf = sp.diags(1.0 / np.maximum(sup, 1)) @ C
    conf = conf.tocsr()
    conf.data[conf.data < min_conf] = 0
    conf.eliminate_zeros()
    n_rules = conf.nnz

    def score(rows):
        if agg == "sum":
            return np.asarray((B[rows] @ conf).todense())
        out = np.zeros((len(rows), B.shape[1]))
        for r, u in enumerate(rows):
            a = B[u].indices
            if len(a):
                out[r] = conf[a].max(axis=0).toarray().ravel()
        return out
    score.n_rules = n_rules
    return score
