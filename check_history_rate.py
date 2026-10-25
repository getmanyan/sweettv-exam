"""Compare two extra history-rate settings with the validation baseline."""
import evaluate
from methods import fit_history


def main():
    _, _, split, labels, users, test_users = evaluate.setup()
    baseline = None
    for smooth, half_life in [(40.0, None), (60.0, None), (40.0, 12)]:
        params = dict(mode="rate", smooth=smooth, half_life=half_life)
        _, result = evaluate.run(
            "history_rate", fit_history, params, split, labels, users, test_users
        )
        if baseline is None:
            baseline = result["map5"]
        delta = result["map5"] - baseline
        print(f"smooth={smooth:g}, half_life={half_life}: "
              f"MAP@5={result['map5']:.5f}, delta={delta:+.5f}")


if __name__ == "__main__":
    main()
