import os
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from lifelines import KaplanMeierFitter

warnings.filterwarnings("ignore")


@dataclass
class KMPlotConfig:
    """Configuration for KM plot generation"""

    num_samples: int = 100
    censoring_rate: float = 0.3

    # Visual parameters
    figure_size: Tuple[int, int] = (10, 6)
    dpi: int = 100
    line_width: float = 2.0

    # Axis parameters
    x_limits: Optional[Tuple[float, float]] = None
    y_limits: Optional[Tuple[float, float]] = (0, 1)
    truncate_y_axis: bool = False
    y_truncate_min: float = 0.0
    y_truncate_max: float = 1.0

    # Grid and styling
    show_grid: bool = True
    grid_alpha: float = 0.3
    background_color: str = "white"

    # Resolution and quality
    add_noise: bool = False
    noise_level: float = 0.1
    blur_kernel: int = 0
    jpeg_quality: int = 95

    # Multiple curves
    num_groups: int = 1
    group_names: Optional[List[str]] = None
    colors: Optional[List[str]] = None

    # Text and annotations
    title: str = "Kaplan-Meier Survival Curve"
    xlabel: str = "Time"
    ylabel: str = "Survival Probability"
    show_legend: bool = True
    font_size: int = 12
    title_font_size: int = 14

    # Risk tables
    show_risk_table: bool = False
    risk_table_height: float = 0.25

    # Confidence intervals
    show_confidence_intervals: bool = False
    ci_alpha: float = 0.3


class BaseKMGenerator(ABC):
    """Base class for all Kaplan-Meier plot generators"""

    def __init__(self, config: KMPlotConfig, random_seed: int = 42):
        self.config = config
        self.random_seed = random_seed
        np.random.seed(random_seed)

    @abstractmethod
    def generate_survival_data(self) -> pd.DataFrame:
        """Generate synthetic survival data"""
        pass

    @abstractmethod
    def get_generator_name(self) -> str:
        """Return the name of this generator"""
        pass

    def apply_visual_effects(self, fig, ax):
        """Apply visual effects like noise, blur, etc."""
        if self.config.add_noise:
            ax.set_facecolor(
                (
                    np.random.normal(1.0, self.config.noise_level * 0.1),
                    np.random.normal(1.0, self.config.noise_level * 0.1),
                    np.random.normal(1.0, self.config.noise_level * 0.1),
                )
            )

        if self.config.truncate_y_axis:
            ax.set_ylim(self.config.y_truncate_min, self.config.y_truncate_max)

    def create_plot(self, data: pd.DataFrame, save_path: str = None) -> Tuple[plt.Figure, plt.Axes]:
        """Create the KM plot from data"""

        fig, ax = plt.subplots(figsize=self.config.figure_size, dpi=self.config.dpi)

        if self.config.num_groups == 1:
            # Single group
            kmf = KaplanMeierFitter()
            kmf.fit(durations=data["time"], event_observed=data["event"])

            color = self.config.colors[0] if self.config.colors else "blue"
            kmf.plot_survival_function(
                ax=ax, linewidth=self.config.line_width, color=color, show_censors=True
            )

            if self.config.show_confidence_intervals:
                ax.fill_between(
                    kmf.confidence_interval_.index,
                    kmf.confidence_interval_.iloc[:, 0],
                    kmf.confidence_interval_.iloc[:, 1],
                    alpha=self.config.ci_alpha,
                    color=color,
                )
        else:
            # Multiple groups
            colors = self.config.colors or sns.color_palette("husl", self.config.num_groups)
            group_names = self.config.group_names or [
                f"Group {i+1}" for i in range(self.config.num_groups)
            ]

            for i in range(self.config.num_groups):
                group_data = data[data["group"] == i]
                kmf = KaplanMeierFitter()
                kmf.fit(
                    durations=group_data["time"],
                    event_observed=group_data["event"],
                    label=group_names[i],
                )

                kmf.plot_survival_function(
                    ax=ax, linewidth=self.config.line_width, color=colors[i], show_censors=True
                )

                if self.config.show_confidence_intervals:
                    ax.fill_between(
                        kmf.confidence_interval_.index,
                        kmf.confidence_interval_.iloc[:, 0],
                        kmf.confidence_interval_.iloc[:, 1],
                        alpha=self.config.ci_alpha,
                        color=colors[i],
                    )

        # Styling
        ax.set_title(self.config.title, fontsize=self.config.title_font_size)
        ax.set_xlabel(self.config.xlabel, fontsize=self.config.font_size)
        ax.set_ylabel(self.config.ylabel, fontsize=self.config.font_size)

        if self.config.show_grid:
            ax.grid(True, alpha=self.config.grid_alpha)

        if self.config.x_limits:
            ax.set_xlim(self.config.x_limits)

        if self.config.y_limits:
            ax.set_ylim(self.config.y_limits)

        if self.config.show_legend and self.config.num_groups > 1:
            ax.legend(fontsize=self.config.font_size)

        self.apply_visual_effects(fig, ax)

        fig.patch.set_facecolor(self.config.background_color)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches="tight")

        return fig, ax

    def generate_and_save(
        self,
        output_dir: str,
        filename_prefix: str = None,
        plots_dir: str = None,
        data_dir: str = None,
        metadata_dir: str = None,
    ) -> Dict:
        """Generate data and save plot with organized folder structure"""
        if filename_prefix is None:
            filename_prefix = self.get_generator_name().lower().replace(" ", "_")

        plots_dir = plots_dir or output_dir
        data_dir = data_dir or output_dir
        metadata_dir = metadata_dir or output_dir

        data = self.generate_survival_data()

        os.makedirs(plots_dir, exist_ok=True)
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(metadata_dir, exist_ok=True)

        data_path = os.path.join(data_dir, f"{filename_prefix}_data.csv")
        data.to_csv(data_path, index=False)

        plot_path = os.path.join(plots_dir, f"{filename_prefix}_plot.png")
        fig, ax = self.create_plot(data, plot_path)

        metadata = {
            "generator": self.get_generator_name(),
            "config": self.config.__dict__,
            "data_path": data_path,
            "plot_path": plot_path,
            "num_samples": len(data),
            "censoring_rate": (1 - data["event"].mean()) if "event" in data.columns else None,
            "time_range": (data["time"].min(), data["time"].max())
            if "time" in data.columns
            else None,
        }

        metadata_path = os.path.join(metadata_dir, f"{filename_prefix}_metadata.json")
        import json

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        plt.close(fig)

        return metadata


def generate_flexible_weibull_survival_times(
    n: int, shape: float = 1.0, scale: float = 10.0, random_seed: int = None
) -> np.ndarray:
    """
    Generate survival times from flexible Weibull distribution

    Parameters:
    - shape (alpha): Controls curve shape
      - shape < 1: Early drop/heavy tail (high early mortality)
      - shape = 1: Exponential (constant hazard)
      - shape > 1: Late drop (increasing hazard over time)
    - scale: Scale parameter (stretch w.r.t to time axis)
    - n: Number of samples
    - random_seed
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    return scale * np.random.weibull(shape, size=n)


def apply_censoring(
    survival_times: np.ndarray,
    censoring_rate: float = 0.3,
    censoring_type: str = "random",
    random_seed: int = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply censoring to survival times"""
    if random_seed is not None:
        np.random.seed(random_seed)

    if censoring_type == "random":
        target_scale = np.mean(survival_times) * (1 - censoring_rate) / censoring_rate
        censoring_times = np.random.exponential(scale=target_scale, size=len(survival_times))
        observed_times = np.minimum(survival_times, censoring_times)
        events = survival_times <= censoring_times

    elif censoring_type == "administrative":
        admin_time = np.percentile(survival_times, 100 * (1 - censoring_rate))
        observed_times = np.minimum(survival_times, admin_time)
        events = survival_times <= admin_time

    elif censoring_type == "informative":
        censoring_prob = 1 / (1 + np.exp(-(survival_times - np.mean(survival_times))))
        censoring_prob = censoring_prob * censoring_rate / np.mean(censoring_prob)
        censor_mask = np.random.binomial(1, censoring_prob, size=len(survival_times))

        observed_times = survival_times.copy()
        events = np.ones_like(survival_times, dtype=bool)

        censored_indices = np.where(censor_mask)[0]
        for idx in censored_indices:
            observed_times[idx] = np.random.uniform(0, survival_times[idx])
            events[idx] = False

    return observed_times, events.astype(int)
