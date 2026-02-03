import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from metrics import BatchMetricsEvaluator, KMMetrics


def load_ground_truth_data(
    plot_name: str, data_dir: str = "data"
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    if plot_name.startswith("plot_"):
        synthetic_data_dir = os.path.join(data_dir, "synthetic_data", "benchmark", "data")
        csv_name = plot_name.replace("_plot", "_data.csv")
        csv_path = os.path.join(synthetic_data_dir, csv_name)

        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)

                duration_col = None
                event_col = None

                for col in df.columns:
                    col_lower = col.lower()
                    if "duration" in col_lower or "time" in col_lower:
                        duration_col = col
                    elif "event" in col_lower or "censored" in col_lower or "status" in col_lower:
                        event_col = col

                if duration_col and event_col:
                    durations = df[duration_col].values
                    events = df[event_col].values

                    if "censored" in event_col.lower():
                        events = 1 - events

                    print(f"Loaded {len(durations)} patients, {events.sum()} events")
                    return durations, events

            except Exception as e:
                print(f"Error loading {csv_path}: {e}")
        else:
            print(f"File not found: {csv_path}")

    print(f"No ground truth found for {plot_name}")
    return None


def load_extracted_data(json_path: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    try:
        with open(json_path, "r") as f:
            data = json.load(f)

        all_times = []
        all_groups = []

        for group_name, points in data.items():
            if not points:
                continue

            times = [p[0] for p in points]
            survival_probs = [p[1] for p in points]

            n_patients = max(10, int(100 * (1 - min(survival_probs))))

            for i in range(len(times) - 1):
                t1, s1 = times[i], survival_probs[i]
                t2, s2 = times[i + 1], survival_probs[i + 1]

                if s1 > s2:
                    n_events = int(n_patients * (s1 - s2))
                    event_times = np.random.uniform(t1, t2, n_events)
                    all_times.extend(event_times)
                    all_groups.extend([group_name] * n_events)

        if not all_times:
            return None

        durations = np.array(all_times)
        events = np.ones(len(durations))

        return durations, events

    except Exception as e:
        print(f"Error loading extracted data from {json_path}: {e}")
        return None


def evaluate_single_plot(plot_dir: str, data_dir: str = "data") -> Optional[Dict]:
    plot_name = os.path.basename(plot_dir)
    json_path = os.path.join(plot_dir, "final_extracted_data.json")

    if not os.path.exists(json_path):
        print(f"No extracted data found for {plot_name}")
        return None

    truth_data = load_ground_truth_data(plot_name, data_dir)
    if truth_data is None:
        return None

    truth_durations, truth_events = truth_data

    extracted_data = load_extracted_data(json_path)
    if extracted_data is None:
        return None

    extracted_durations, extracted_events = extracted_data

    metrics = KMMetrics()
    try:
        results = metrics.evaluate_reconstruction(
            truth_durations, truth_events, extracted_durations, extracted_events
        )
        results["plot_name"] = plot_name
        return results
    except Exception as e:
        print(f"Error evaluating {plot_name}: {e}")
        return None


def evaluate_logs_directory(logs_dir: str, data_dir: str = "data") -> Dict:
    plot_dirs = [d for d in os.listdir(logs_dir) if os.path.isdir(os.path.join(logs_dir, d))]

    evaluator = BatchMetricsEvaluator()
    successful_evaluations = 0
    failed_evaluations = 0

    for plot_dir in plot_dirs:
        plot_path = os.path.join(logs_dir, plot_dir)
        print(f"\nEvaluating plot: {plot_dir}")

        result = evaluate_single_plot(plot_path, data_dir)
        if result is not None:
            plot_name = result["plot_name"]

            if "dataset_id" not in result:
                result["dataset_id"] = plot_name

            evaluator.results.append(result)
            successful_evaluations += 1

            print(f"IAE: {result['integrated_absolute_error']:.6f}")
            if result["median_os_error"] is not None:
                print(f"Median OS Error: {result['median_os_error']:.6f}")
        else:
            failed_evaluations += 1
            print(f"Evaluation failed")

    print(f"\nEvaluation Summary:")
    print(f"  Successful: {successful_evaluations}")
    print(f"  Failed: {failed_evaluations}")

    if successful_evaluations > 0:
        summary = evaluator.compute_summary_statistics()
        results_df = evaluator.export_results_to_dataframe()

        return {
            "summary_statistics": summary,
            "individual_results": evaluator.results,
            "results_dataframe": results_df,
            "successful_evaluations": successful_evaluations,
            "failed_evaluations": failed_evaluations,
        }
    else:
        return {
            "summary_statistics": {},
            "individual_results": [],
            "results_dataframe": pd.DataFrame(),
            "successful_evaluations": successful_evaluations,
            "failed_evaluations": failed_evaluations,
        }


def main():
    parser = argparse.ArgumentParser(description="Evaluate KM digitization pipeline performance")
    parser.add_argument("logs_dir", help="Directory containing extraction logs")
    parser.add_argument("--data-dir", default="data", help="Base data directory (default: data)")
    parser.add_argument("--output", "-o", help="Output file for results (JSON format)")

    args = parser.parse_args()

    if not os.path.exists(args.logs_dir):
        print(f"Logs directory not found: {args.logs_dir}")
        return

    results = evaluate_logs_directory(args.logs_dir, args.data_dir)

    summary = results["summary_statistics"]
    if summary:
        print(f"\nFINAL RESULTS:")
        print(f"{'='*60}")

        iae_stats = summary.get("iae", {})
        if iae_stats:
            print(f"Integrated Absolute Error (IAE):")
            print(f"  Median: {iae_stats.get('median', 0):.6f}")
            ci_95 = iae_stats.get("ci_95", (0, 0))
            print(f"  95% CI: {ci_95[0]:.6f} - {ci_95[1]:.6f}")
            print(f"  Mean ± SD: {iae_stats.get('mean', 0):.6f} ± {iae_stats.get('std', 0):.6f}")

        median_ae_stats = summary.get("median_ae", {})
        if median_ae_stats:
            print(f"Median Absolute Error:")
            print(f"  Median: {median_ae_stats.get('median', 0):.6f}")
            ci_95 = median_ae_stats.get("ci_95", (0, 0))
            print(f"  95% CI: {ci_95[0]:.6f} - {ci_95[1]:.6f}")

        median_os_stats = summary.get("median_os_error", {})
        if median_os_stats:
            print(f"Median OS Error:")
            print(f"  Median: {median_os_stats.get('median', 0):.6f}")
            ci_95 = median_os_stats.get("ci_95", (0, 0))
            print(f"  95% CI: {ci_95[0]:.6f} - {ci_95[1]:.6f}")
            print(f"  Available for {median_os_stats.get('n_datasets_with_median', 0)} datasets")

        print(f"{'='*60}")

    if args.output:
        output_data = {
            "summary_statistics": summary,
            "individual_results": [
                {
                    k: v
                    for k, v in result.items()
                    if k not in ["truth_curve", "reconstructed_curve", "absolute_errors"]
                }
                for result in results["individual_results"]
            ],
            "successful_evaluations": results["successful_evaluations"],
            "failed_evaluations": results["failed_evaluations"],
        }

        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2, default=str)
        print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
