import argparse
import json
from pathlib import Path

from runtime_paths import CHECKPOINT_DIR, CHECKPOINT_NC2023_DIR, CHECKPOINT_OTW_DIR


STEP_ORDER = ["1d", "3d", "5d", "7d"]


def parse_args():
    parser = argparse.ArgumentParser(description="Compare soil-moisture forecasting experiment results")
    parser.add_argument("--baseline-dir", type=str, default=str(CHECKPOINT_DIR))
    parser.add_argument("--lss-dir", type=str, default=str(CHECKPOINT_NC2023_DIR))
    parser.add_argument("--otw-dir", type=str, default=str(CHECKPOINT_OTW_DIR))
    return parser.parse_args()


def load_results(run_dir):
    run_dir = Path(run_dir)
    results_path = run_dir / "test_results.json"
    if not results_path.exists():
        return None

    with open(results_path, "r", encoding="utf-8") as f:
        return json.load(f)


def format_metric(value):
    return f"{value:.4f}" if value is not None else "-"


def print_primary_table(run_results):
    print("7d primary table")
    print("| Method | RMSE | MAE | R2 |")
    print("| --- | ---: | ---: | ---: |")

    for method_name, results in run_results.items():
        if results is None:
            continue
        metrics = results.get("Steps", {}).get("7d")
        if metrics is None:
            continue
        print(
            f"| {method_name} | "
            f"{format_metric(metrics.get('RMSE'))} | "
            f"{format_metric(metrics.get('MAE'))} | "
            f"{format_metric(metrics.get('R2'))} |"
        )


def print_all_horizon_table(run_results):
    print("\nAll-horizon summary")
    print("| Method | Horizon | RMSE | MAE | R2 |")
    print("| --- | --- | ---: | ---: | ---: |")

    for method_name, results in run_results.items():
        if results is None:
            continue
        step_results = results.get("Steps", {})
        for step in STEP_ORDER:
            metrics = step_results.get(step)
            if metrics is None:
                continue
            print(
                f"| {method_name} | {step} | "
                f"{format_metric(metrics.get('RMSE'))} | "
                f"{format_metric(metrics.get('MAE'))} | "
                f"{format_metric(metrics.get('R2'))} |"
            )


def print_delta_table(run_results):
    baseline = run_results.get("Baseline")
    if baseline is None:
        return

    baseline_steps = baseline.get("Steps", {})
    print("\nDelta vs baseline")
    print("| Method | Horizon | dRMSE | dMAE | dR2 |")
    print("| --- | --- | ---: | ---: | ---: |")

    for method_name, results in run_results.items():
        if method_name == "Baseline" or results is None:
            continue

        step_results = results.get("Steps", {})
        for step in STEP_ORDER:
            base_metrics = baseline_steps.get(step)
            metrics = step_results.get(step)
            if base_metrics is None or metrics is None:
                continue

            d_rmse = metrics["RMSE"] - base_metrics["RMSE"]
            d_mae = metrics["MAE"] - base_metrics["MAE"]
            d_r2 = metrics["R2"] - base_metrics["R2"]
            print(f"| {method_name} | {step} | {d_rmse:+.4f} | {d_mae:+.4f} | {d_r2:+.4f} |")


def main():
    args = parse_args()

    run_results = {
        "Baseline": load_results(args.baseline_dir),
        "LSS": load_results(args.lss_dir),
        "OTW": load_results(args.otw_dir),
    }

    available = [name for name, result in run_results.items() if result is not None]
    if not available:
        print("No result files found.")
        return

    print(f"Loaded results: {', '.join(available)}")
    print_primary_table(run_results)
    print_all_horizon_table(run_results)
    print_delta_table(run_results)


if __name__ == "__main__":
    main()
