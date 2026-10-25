"""Build user-programme interactions from the sweet.tv view log and broadcast schedule.

Matching rule (mirrors how the competition built its labels, see the Kaggle Data tab):
  1. Drop schedule rows with tv_show_id == 0 (unlabelled airtime, not an item).
  2. A view row matches an airing when it is on the same channel (vsetv_id == channel_id)
     and the view interval [start_time, stop_time) overlaps the airing interval
     [start_time, start_time + duration).
  3. For each (user, airing) the overlapping seconds are summed over all view rows,
     then divided by the airing's duration -> screen time.
  4. An airing counts as "watched" when screen time >= WATCH_THRESHOLD (0.8 by default).
Output: one row per (user, airing) with overlap seconds, screen time and the watched flag,
plus the labelled airing table with metadata. Both are saved as pickles in OUT_DIR.

Usage: python prepare.py DATA_DIR OUT_DIR [--threshold 0.8]
"""
import argparse
import os
import time

import numpy as np
import pandas as pd

VIEW_FILE = "dataset11-30.csv"
SCHEDULE_FILES = [
    "export_arh_11-20-final.csv",
    "export_arh_21-30-final.csv",
    "export_arh_31-42-final.csv",
]
META_COLS = [
    "tv_show_title", "tv_show_category", "tv_show_genre_1", "tv_show_genre_2",
    "tv_show_genre_3", "year_of_production", "director", "actors",
]
CHANNEL_SPAN = 10 ** 10  # larger than any unix timestamp in seconds, used for composite keys


def to_unix_seconds(col: pd.Series, fmt: str) -> np.ndarray:
    """Parse naive local timestamps with an explicit format (schedule is day-first: 09.03.2020)."""
    ts = pd.to_datetime(col, format=fmt, errors="coerce")
    n_bad = int(ts.isna().sum())
    if n_bad:
        print(f"  warning: {n_bad:,} unparseable timestamps")
    return (ts.astype("datetime64[s]").astype("int64")).to_numpy()


SCHEDULE_TIME_FMT = "%d.%m.%Y %H:%M:%S"
VIEW_TIME_FMT = "%Y-%m-%d %H:%M:%S"


def find_col(df: pd.DataFrame, *names: str) -> str:
    for n in names:
        if n in df.columns:
            return n
    raise KeyError(f"none of {names} in {list(df.columns)}")


def load_schedule(data_dir: str) -> pd.DataFrame:
    parts = []
    for f in SCHEDULE_FILES:
        df = pd.read_csv(os.path.join(data_dir, f), low_memory=False)
        df["source_file"] = f
        parts.append(df)
    sch = pd.concat(parts, ignore_index=True)
    n_all = len(sch)
    sch = sch[sch["tv_show_id"].fillna(0).astype("int64") != 0].copy()
    sch["tv_show_id"] = sch["tv_show_id"].astype("int64")
    sch["air_start"] = to_unix_seconds(sch["start_time"], SCHEDULE_TIME_FMT)
    sch["duration"] = pd.to_numeric(sch["duration"], errors="coerce")
    sch = sch[sch["duration"] > 0]
    sch["air_end"] = sch["air_start"] + sch["duration"].astype("int64")
    sch = sch.drop_duplicates(["channel_id", "air_start", "tv_show_id"])
    sch = sch.sort_values(["channel_id", "air_start"]).reset_index(drop=True)
    # Airings on one channel should not overlap; clip any overlap so end keys stay sorted.
    nxt = sch.groupby("channel_id")["air_start"].shift(-1)
    sch["join_end"] = np.where(nxt.notna(), np.minimum(sch["air_end"], nxt.fillna(0)), sch["air_end"]).astype("int64")
    sch["airing_id"] = np.arange(len(sch), dtype="int64")
    print(f"schedule rows {n_all:,}; labelled airings kept {len(sch):,} ({len(sch) / n_all:.1%})")
    return sch


def load_views(data_dir: str) -> pd.DataFrame:
    # user_id values exceed int64, so keep them as strings end to end
    v = pd.read_csv(os.path.join(data_dir, VIEW_FILE), dtype={"user_id": "str"}, low_memory=False)
    ch = find_col(v, "vsetv_id", "channel_id")
    st = find_col(v, "start_time")
    en = find_col(v, "stop_time", "end_time")
    out = pd.DataFrame({
        "user_id": v["user_id"].astype("str"),
        "channel_id": v[ch],
        "v_start": to_unix_seconds(v[st], VIEW_TIME_FMT),
        "v_end": to_unix_seconds(v[en], VIEW_TIME_FMT),
    })
    out = out[out["v_end"] > out["v_start"]].reset_index(drop=True)
    print(f"view rows {len(v):,}; valid intervals {len(out):,}")
    return out


def join(views: pd.DataFrame, sch: pd.DataFrame) -> pd.DataFrame:
    channels = pd.Index(pd.unique(pd.concat([sch["channel_id"], views["channel_id"]])))
    sch_ch = channels.get_indexer(sch["channel_id"]).astype("int64")
    v_ch = channels.get_indexer(views["channel_id"]).astype("int64")
    a_start_k = sch_ch * CHANNEL_SPAN + sch["air_start"].to_numpy()
    a_end_k = sch_ch * CHANNEL_SPAN + sch["join_end"].to_numpy()
    assert np.all(np.diff(a_start_k) >= 0) and np.all(np.diff(a_end_k) >= 0)
    s_k = v_ch * CHANNEL_SPAN + views["v_start"].to_numpy()
    e_k = v_ch * CHANNEL_SPAN + views["v_end"].to_numpy()
    lo = np.searchsorted(a_end_k, s_k, side="right")   # first airing ending after the view starts
    hi = np.searchsorted(a_start_k, e_k, side="left")  # airings starting before the view ends
    cnt = np.clip(hi - lo, 0, None)
    v_idx = np.repeat(np.arange(len(views)), cnt)
    offs = np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt)
    a_idx = np.repeat(lo, cnt) + offs
    a_start = sch["air_start"].to_numpy()[a_idx]
    a_end = sch["air_end"].to_numpy()[a_idx]
    ov = np.minimum(views["v_end"].to_numpy()[v_idx], a_end) - np.maximum(views["v_start"].to_numpy()[v_idx], a_start)
    keep = ov > 0
    pairs = pd.DataFrame({
        "user_id": views["user_id"].to_numpy()[v_idx[keep]],
        "airing_id": a_idx[keep],
        "overlap": ov[keep],
    })
    ua = pairs.groupby(["user_id", "airing_id"], sort=False)["overlap"].sum().reset_index()
    print(f"view-airing overlaps {keep.sum():,}; user-airing pairs {len(ua):,}")
    return ua


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--threshold", type=float, default=0.8)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    t0 = time.time()
    sch = load_schedule(args.data_dir)
    iso = pd.to_datetime(sch["air_start"], unit="s").dt.isocalendar()
    sch["week"] = iso["week"].astype("int64").to_numpy()
    views = load_views(args.data_dir)
    ua = join(views, sch)
    ua = ua.merge(sch[["airing_id", "tv_show_id", "duration", "week", "air_start"]], on="airing_id")
    ua["screen"] = ua["overlap"] / ua["duration"]
    ua["watched"] = ua["screen"] >= args.threshold
    print(f"watched airings (screen >= {args.threshold}): {ua['watched'].sum():,} of {len(ua):,}")
    sch.to_pickle(os.path.join(args.out_dir, "airings.pkl"))
    ua.to_pickle(os.path.join(args.out_dir, "user_airings.pkl"))
    print(f"done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
