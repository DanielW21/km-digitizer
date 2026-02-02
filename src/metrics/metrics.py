from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter


class KMMetrics:
    """Class for computing Kaplan-Meier reconstruction metrics."""

    def __init__(self):
        self.km_fitter = KaplanMeierFitter()

    def normalize_time(self, times: np.ndarray, max_time: float) -> np.ndarray:
        return times / max_time

    def interpolate_survival_curve(
        self, times: np.ndarray, survival_probs: np.ndarray, target_times: np.ndarray
    ) -> np.ndarray:
        return np.interp(target_times, times, survival_probs)

    def compute_absolute_error(
        self,
        truth_times: np.ndarray,
        truth_survival: np.ndarray,
        reconstructed_times: np.ndarray,
        reconstructed_survival: np.ndarray,
        normalize: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if normalize:
            max_time = max(np.max(truth_times), np.max(reconstructed_times))
            truth_times_norm = self.normalize_time(truth_times, max_time)
            reconstructed_times_norm = self.normalize_time(reconstructed_times, max_time)
        else:
            truth_times_norm = truth_times
            reconstructed_times_norm = reconstructed_times

        min_time = max(np.min(truth_times_norm), np.min(reconstructed_times_norm))
        max_time_common = min(np.max(truth_times_norm), np.max(reconstructed_times_norm))
        common_times = np.linspace(min_time, max_time_common, 1000)

        truth_interp = self.interpolate_survival_curve(
            truth_times_norm, truth_survival, common_times
        )
        reconstructed_interp = self.interpolate_survival_curve(
            reconstructed_times_norm, reconstructed_survival, common_times
        )

        absolute_errors = np.abs(truth_interp - reconstructed_interp)

        return common_times, absolute_errors

    def compute_integrated_absolute_error(
        self,
        truth_times: np.ndarray,
        truth_survival: np.ndarray,
        reconstructed_times: np.ndarray,
        reconstructed_survival: np.ndarray,
        normalize: bool = True,
    ) -> float:
        common_times, absolute_errors = self.compute_absolute_error(
            truth_times, truth_survival, reconstructed_times, reconstructed_survival, normalize
        )
        iae = np.trapezoid(absolute_errors, common_times)
        return iae

    def compute_median_survival(
        self, times: np.ndarray, survival_probs: np.ndarray
    ) -> Optional[float]:
        """
        Compute median survival time from survival curve.

        Args:
            times: Time points
            survival_probs: Survival probabilities

        Returns:
            Median survival time (time when survival probability = 0.5)
        """
        # Find the time point where survival probability crosses 0.5
        if np.min(survival_probs) > 0.5:
            # Median not reached
            return None

        # Find first index where survival drops to or below 0.5
        idx = np.where(survival_probs <= 0.5)[0]
        if len(idx) == 0:
            return None

        first_idx = idx[0]

        if first_idx == 0:
            return times[0]

        # Linear interpolation between the two points
        t1, s1 = times[first_idx - 1], survival_probs[first_idx - 1]
        t2, s2 = times[first_idx], survival_probs[first_idx]

        # Interpolate to find exact time when survival = 0.5
        median_time = t1 + (0.5 - s1) * (t2 - t1) / (s2 - s1)

        return median_time

    def compute_median_os_error(
        self,
        truth_times: np.ndarray,
        truth_survival: np.ndarray,
        reconstructed_times: np.ndarray,
        reconstructed_survival: np.ndarray,
        normalize: bool = True,
    ) -> Optional[float]:
        if normalize:
            max_time = max(np.max(truth_times), np.max(reconstructed_times))
            truth_times_norm = self.normalize_time(truth_times, max_time)
            reconstructed_times_norm = self.normalize_time(reconstructed_times, max_time)
        else:
            truth_times_norm = truth_times
            reconstructed_times_norm = reconstructed_times

        truth_median = self.compute_median_survival(truth_times_norm, truth_survival)
        reconstructed_median = self.compute_median_survival(
            reconstructed_times_norm, reconstructed_survival
        )

        if truth_median is None or reconstructed_median is None:
            return None

        return abs(truth_median - reconstructed_median)

    def fit_kaplan_meier_from_ipd(
        self, durations: np.ndarray, events: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        self.km_fitter.fit(durations, events)
        return self.km_fitter.timeline, self.km_fitter.survival_function_.values.flatten()

    def evaluate_reconstruction(
        self,
        truth_durations: np.ndarray,
        truth_events: np.ndarray,
        reconstructed_durations: np.ndarray,
        reconstructed_events: np.ndarray,
    ) -> Dict:
        truth_times, truth_survival = self.fit_kaplan_meier_from_ipd(truth_durations, truth_events)
        reconstructed_times, reconstructed_survival = self.fit_kaplan_meier_from_ipd(
            reconstructed_durations, reconstructed_events
        )

        common_times, ae_values = self.compute_absolute_error(
            truth_times, truth_survival, reconstructed_times, reconstructed_survival
        )

        iae = self.compute_integrated_absolute_error(
            truth_times, truth_survival, reconstructed_times, reconstructed_survival
        )

        median_os_error = self.compute_median_os_error(
            truth_times, truth_survival, reconstructed_times, reconstructed_survival
        )

        results = {
            "absolute_errors": {
                "times": common_times,
                "errors": ae_values,
                "median_ae": np.median(ae_values),
                "mean_ae": np.mean(ae_values),
                "q95_ae": np.percentile(ae_values, 95),
            },
            "integrated_absolute_error": iae,
            "median_os_error": median_os_error,
            "truth_curve": {
                "times": truth_times,
                "survival": truth_survival,
                "median_os": self.compute_median_survival(truth_times, truth_survival),
            },
            "reconstructed_curve": {
                "times": reconstructed_times,
                "survival": reconstructed_survival,
                "median_os": self.compute_median_survival(
                    reconstructed_times, reconstructed_survival
                ),
            },
        }

        return results


class BatchMetricsEvaluator:
    def __init__(self):
        self.metrics = KMMetrics()
        self.results = []

    def evaluate_dataset(
        self,
        dataset_id: str,
        truth_durations: np.ndarray,
        truth_events: np.ndarray,
        reconstructed_durations: np.ndarray,
        reconstructed_events: np.ndarray,
    ) -> Dict:
        results = self.metrics.evaluate_reconstruction(
            truth_durations, truth_events, reconstructed_durations, reconstructed_events
        )
        results["dataset_id"] = dataset_id
        self.results.append(results)
        return results

    def compute_summary_statistics(self) -> Dict:
        if not self.results:
            return {}

        median_aes = [r["absolute_errors"]["median_ae"] for r in self.results]
        iaes = [r["integrated_absolute_error"] for r in self.results]
        median_os_errors = [
            r["median_os_error"] for r in self.results if r["median_os_error"] is not None
        ]

        summary = {
            "n_datasets": len(self.results),
            "median_ae": {
                "median": np.median(median_aes),
                "ci_95": (np.percentile(median_aes, 2.5), np.percentile(median_aes, 97.5)),
                "mean": np.mean(median_aes),
                "std": np.std(median_aes),
            },
            "iae": {
                "median": np.median(iaes),
                "ci_95": (np.percentile(iaes, 2.5), np.percentile(iaes, 97.5)),
                "mean": np.mean(iaes),
                "std": np.std(iaes),
            },
        }

        if median_os_errors:
            summary["median_os_error"] = {
                "median": np.median(median_os_errors),
                "ci_95": (
                    np.percentile(median_os_errors, 2.5),
                    np.percentile(median_os_errors, 97.5),
                ),
                "mean": np.mean(median_os_errors),
                "std": np.std(median_os_errors),
                "n_datasets_with_median": len(median_os_errors),
            }

        return summary

    def export_results_to_dataframe(self) -> pd.DataFrame:
        if not self.results:
            return pd.DataFrame()

        data = []
        for result in self.results:
            row = {
                "dataset_id": result["dataset_id"],
                "median_ae": result["absolute_errors"]["median_ae"],
                "mean_ae": result["absolute_errors"]["mean_ae"],
                "q95_ae": result["absolute_errors"]["q95_ae"],
                "iae": result["integrated_absolute_error"],
                "median_os_error": result["median_os_error"],
                "truth_median_os": result["truth_curve"]["median_os"],
                "reconstructed_median_os": result["reconstructed_curve"]["median_os"],
            }
            data.append(row)

        return pd.DataFrame(data)


def compute_performance_by_parameters(
    results_df: pd.DataFrame, parameter_columns: List[str]
) -> pd.DataFrame:
    if results_df.empty:
        return pd.DataFrame()

    grouped = (
        results_df.groupby(parameter_columns)
        .agg(
            {
                "median_ae": ["count", "median", "mean", "std"],
                "iae": ["median", "mean", "std"],
                "median_os_error": ["median", "mean", "std", "count"],
            }
        )
        .round(6)
    )

    grouped.columns = ["_".join(col).strip() for col in grouped.columns.values]
    grouped.reset_index(inplace=True)

    return grouped


if __name__ == "__main__":
    evaluator = BatchMetricsEvaluator()
    np.random.seed(42)

    for i in range(3):
        n_patients = 50
        truth_times = np.random.exponential(30, n_patients)
        truth_events = np.random.binomial(1, 0.7, n_patients)

        recon_times = truth_times + np.random.normal(0, 1, n_patients)
        recon_times = np.maximum(recon_times, 0.1)
        recon_events = truth_events.copy()

        graph_id = f"graph_{i+1:03d}"
        evaluator.evaluate_dataset(graph_id, truth_times, truth_events, recon_times, recon_events)

    summary = evaluator.compute_summary_statistics()
    print("\nResults Summary:")
    print(
        f"IAE: {summary['iae']['median']:.6f} (95% CI: {summary['iae']['ci_95'][0]:.6f}-{summary['iae']['ci_95'][1]:.6f})"
    )

    results_df = evaluator.export_results_to_dataframe()
    cols = ["dataset_id", "iae", "median_ae"]
    print(f"\nDetailed Results:\n{results_df[cols].round(6)}")
