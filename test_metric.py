"""Checks common.apk against the instructor's worked MAP@5 example (relevant = A, B, C)."""
from common import apk, mapk

rel = ["A", "B", "C"]
x = apk(rel, ["A", "B", "D", "C", "E"])   # expected (1.00 + 1.00 + 0.75) / 3 = 0.9167
y = apk(rel, ["D", "A", "E", "B", "C"])   # expected (0.50 + 0.50 + 0.60) / 3 = 0.5333
print(f"Method X AP@5 = {x:.4f}  (slide: 0.917)")
print(f"Method Y AP@5 = {y:.4f}  (slide: 0.533)")
assert abs(x - 2.75 / 3) < 1e-9 and abs(y - 1.6 / 3) < 1e-9
# in this competition every user has exactly 5 relevant shows, so the denominator is always min(5, 5) = 5
assert abs(apk([1, 2, 3, 4, 5], [9, 1, 9, 2, 9]) - (1 / 2 + 2 / 4) / 5) < 1e-9
print("MAP over both =", round(mapk({"x": rel, "y": rel}, {"x": ["A", "B", "D", "C", "E"], "y": ["D", "A", "E", "B", "C"]}), 4))
print("all metric checks passed")
