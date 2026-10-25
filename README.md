# sweet.tv TV programme recommender — LLM4Rec exam (Anastasia Getman)

Top-5 TV programmes per user for the Kaggle competition
[sweet.tv — TV program recommender](https://www.kaggle.com/competitions/sweettv-tv-program-recommender) (metric MAP@5).

## What is here

| File | Role |
|---|---|
| `prepare.py` | Joins the view log with the broadcast schedule → user × airing interactions |
| `common.py` | Temporal split, label construction (competition recipe), MAP@5, candidate pool, fallback |
| `methods.py` | Popularity, personal history, Personalized PageRank, implicit ALS, content-based filtering, association rules |
| `evaluate.py` | One validation protocol for every method → `results/val_results.csv` |
| `tune.py` | Hyperparameter grids on the validation split → `results/val_grid.csv` |
| `hybrid.py` | Score-fusion hybrid (`fusion`) and a switching rule (`switch`), tuned on half A of users, reported on half B |
| `make_submission.py` | Refits on weeks 11–30, predicts weeks 31–42, writes and checks the Kaggle file |
| `examples.py` | Two example users, every method's Top-5 with titles and categories, AP@5 hand-check sheet |
| `explore.py` | Data checks used in the report (coverage, duration distributions, cold start) |
| `stats.py` | Repeat share, catalogue coverage and MAP@5 by history size → `results/stats.md` |
| `test_metric.py` | Checks the MAP@5 implementation against the instructor's worked example |
| `results/` | All validation outputs used in the report |
| `submissions/` | Kaggle submission files (`fusion.csv` = designated final unless stated otherwise in the report) |

## Key decisions

- **Matching rule.** A view matches an airing when it is on the same channel (`vsetv_id = channel_id`) and the view interval overlaps the airing interval. Overlapping seconds are summed per (user, airing) and divided by the airing's `duration`. An airing counts as *watched* when this screen time is ≥ 0.8 — the same rule the organisers used to build the test labels (Kaggle Data tab).
- **`tv_show_id = 0`.** Dropped: unlabelled airtime is not an item. Only 11.1% of schedule rows carry an id.
- **Validation.** History weeks 11–18 → targets weeks 19–30 (12 weeks, same horizon as the test window 31–42). Targets are built with the competition recipe: per user, the 5 programmes with the most watched airings; users with fewer than 5 such programmes are not evaluated (the competition does the same).
- **Candidate pool.** Programmes that air in the target window (from the schedule only). Already-watched programmes are allowed — identical for every method.
- **Cold start.** Users with no watched airing in history get the popularity list (most distinct viewers in history, restricted to programmes airing in the target window). Same fallback everywhere; it also fills empty slots.
- **Exactly five ids.** Each method's scores are sorted within the candidate pool; only positive scores are used, the remaining slots are filled from the fallback list without duplicates.
- **No future viewing.** Test-window views are never used; the test-window schedule is used only to know what airs when.

## Reproduce

Python ≥ 3.10.

```bash
pip install -r requirements.txt
# download and unzip the Kaggle data into ./data  (5 csv files)
python prepare.py data work          # ~25 s: join → work/airings.pkl, work/user_airings.pkl
python test_metric.py                # metric check against the instructor's example
python explore.py                    # data checks
python evaluate.py                   # all methods, one protocol → results/val_results.csv  (~30 s)
python tune.py history ppr als cbf assoc_rules   # optional, full grids (~15 min)
python hybrid.py fusion              # fusion weights on half A, report on half B (~10 min)
python hybrid.py switch              # switching rule experiment (~1 min)
python examples.py                   # results/examples.md, results/handcheck.md
python stats.py                      # results/stats.md (~1 min)
python make_submission.py fusion     # submissions/fusion.csv
python make_submission.py single history   # submissions/history.csv
```

All randomness (ALS initialisation, user halves) uses seed 42. Reference environment: Python 3.13.16, pandas 3.0.5,
numpy 2.5.3, scipy 1.18.1, scikit-learn 1.9.1; Linux, 2 vCPU, 7 GB RAM.
