from typing import List

import numpy as np
import pandas as pd
from base import (
    BaseKMGenerator,
    KMPlotConfig,
    apply_censoring,
    generate_flexible_weibull_survival_times,
)


class StandardKMGenerator(BaseKMGenerator):
    """Generates standard single-group KM plots with flexible Weibull shapes"""

    def __init__(
        self, config: KMPlotConfig, shape: float = 1.0, scale: float = 10.0, random_seed: int = 42
    ):
        super().__init__(config, random_seed)
        self.shape = shape  # Weibull shape parameter (alpha)
        self.scale = scale

    def generate_survival_data(self) -> pd.DataFrame:
        """Generate survival data using flexible Weibull distribution"""
        survival_times = generate_flexible_weibull_survival_times(
            self.config.num_samples, self.shape, self.scale, self.random_seed
        )

        observed_times, events = apply_censoring(
            survival_times, self.config.censoring_rate, "random", self.random_seed
        )

        return pd.DataFrame({"time": observed_times, "event": events})

    def get_generator_name(self) -> str:
        if self.shape < 1:
            shape_desc = "Early Drop"
        elif self.shape == 1:
            shape_desc = "Exponential"
        else:
            shape_desc = "Late Drop"
        return f"Standard KM ({shape_desc}, α={self.shape:.1f})"


class MultiGroupKMGenerator(BaseKMGenerator):
    """Generates KM plots with multiple overlapping groups using flexible Weibull"""

    def __init__(
        self,
        config: KMPlotConfig,
        group_effects: List[float] = None,
        shapes: List[float] = None,
        random_seed: int = 42,
    ):
        super().__init__(config, random_seed)
        self.group_effects = group_effects or [1.0, 0.7, 1.3][: config.num_groups]
        self.shapes = shapes or [0.7, 1.0, 1.5][: config.num_groups]

        if len(self.group_effects) != config.num_groups:
            raise ValueError("Number of group effects must match number of groups")
        if len(self.shapes) != config.num_groups:
            self.shapes = [1.0] * config.num_groups

    def generate_survival_data(self) -> pd.DataFrame:
        """Generate survival data for multiple groups using flexible Weibull"""
        all_data = []

        samples_per_group = self.config.num_samples // self.config.num_groups

        for group_idx in range(self.config.num_groups):
            # Adjust survival times based on group effect and shape
            base_scale = 10.0
            group_scale = base_scale * self.group_effects[group_idx]

            survival_times = generate_flexible_weibull_survival_times(
                samples_per_group, self.shapes[group_idx], group_scale, self.random_seed + group_idx
            )

            observed_times, events = apply_censoring(
                survival_times, self.config.censoring_rate, "random", self.random_seed + group_idx
            )

            group_data = pd.DataFrame({"time": observed_times, "event": events, "group": group_idx})

            all_data.append(group_data)

        return pd.concat(all_data, ignore_index=True)

    def get_generator_name(self) -> str:
        return f"Multi-Group KM ({self.config.num_groups} groups)"


class EarlyDropoffKMGenerator(BaseKMGenerator):
    """Generates KM plots with early steep dropoff using flexible Weibull with small shape"""

    def __init__(
        self, config: KMPlotConfig, shape: float = 0.5, scale: float = 8.0, random_seed: int = 42
    ):
        super().__init__(config, random_seed)
        self.shape = shape  # Small shape parameter for early dropoff
        self.scale = scale

    def generate_survival_data(self) -> pd.DataFrame:
        """Generate survival data with early dropoff pattern using Weibull shape < 1"""
        survival_times = generate_flexible_weibull_survival_times(
            self.config.num_samples, self.shape, self.scale, self.random_seed
        )

        observed_times, events = apply_censoring(
            survival_times, self.config.censoring_rate, "random", self.random_seed
        )

        return pd.DataFrame({"time": observed_times, "event": events})

    def get_generator_name(self) -> str:
        return f"Early Dropoff KM (\u03b1={self.shape})"


class PlateauKMGenerator(BaseKMGenerator):
    """Generates KM plots with plateau (cure fraction) using flexible Weibull"""

    def __init__(
        self,
        config: KMPlotConfig,
        cure_fraction: float = 0.2,
        susceptible_shape: float = 1.2,
        susceptible_scale: float = 8.0,
        random_seed: int = 42,
    ):
        super().__init__(config, random_seed)
        self.cure_fraction = cure_fraction
        self.susceptible_shape = susceptible_shape
        self.susceptible_scale = susceptible_scale

    def generate_survival_data(self) -> pd.DataFrame:
        """Generate survival data with cure fraction plateau"""
        n_cured = int(self.config.num_samples * self.cure_fraction)
        n_susceptible = self.config.num_samples - n_cured

        # Susceptible population - flexible Weibull with moderate shape
        susceptible_times = generate_flexible_weibull_survival_times(
            n_susceptible, self.susceptible_shape, self.susceptible_scale, self.random_seed
        )

        # Cured population - very long survival times
        cured_times = np.random.exponential(100.0, size=n_cured) + 50.0

        all_times = np.concatenate([susceptible_times, cured_times])
        np.random.shuffle(all_times)

        observed_times, events = apply_censoring(
            all_times, self.config.censoring_rate, "administrative", self.random_seed
        )

        return pd.DataFrame({"time": observed_times, "event": events})

    def get_generator_name(self) -> str:
        return f"Plateau KM (cure={self.cure_fraction:.1%})"


class CrossingCurvesKMGenerator(BaseKMGenerator):
    """Generates KM plots where survival curves cross over time"""

    def __init__(self, config: KMPlotConfig, crossover_time: float = 10.0, random_seed: int = 42):
        super().__init__(config, random_seed)
        self.crossover_time = crossover_time

        if config.num_groups < 2:
            raise ValueError("Crossing curves requires at least 2 groups")


class CrossingCurvesKMGenerator(BaseKMGenerator):
    """Generates KM plots where survival curves cross over time using flexible Weibull"""

    def __init__(
        self,
        config: KMPlotConfig,
        crossover_time: float = 10.0,
        early_shapes: List[float] = None,
        late_shapes: List[float] = None,
        random_seed: int = 42,
    ):
        super().__init__(config, random_seed)
        self.crossover_time = crossover_time
        self.early_shapes = early_shapes or [1.5, 0.7]  # Group 0 better early, Group 1 worse
        self.late_shapes = late_shapes or [0.8, 1.8]  # Group 0 worse late, Group 1 better

        if config.num_groups < 2:
            raise ValueError("Crossing curves requires at least 2 groups")

    def generate_survival_data(self) -> pd.DataFrame:
        """Generate survival data where curves cross"""
        all_data = []
        samples_per_group = self.config.num_samples // self.config.num_groups

        for group_idx in range(self.config.num_groups):
            # Use different shapes for early vs late survival
            early_times = generate_flexible_weibull_survival_times(
                samples_per_group // 2,
                self.early_shapes[min(group_idx, len(self.early_shapes) - 1)],
                8.0,
                self.random_seed + group_idx,
            )

            late_times = (
                generate_flexible_weibull_survival_times(
                    samples_per_group - samples_per_group // 2,
                    self.late_shapes[min(group_idx, len(self.late_shapes) - 1)],
                    12.0,
                    self.random_seed + group_idx + 100,
                )
                + self.crossover_time
            )

            survival_times = np.concatenate([early_times, late_times])
            np.random.shuffle(survival_times)

            observed_times, events = apply_censoring(
                survival_times, self.config.censoring_rate, "random", self.random_seed + group_idx
            )

            group_data = pd.DataFrame({"time": observed_times, "event": events, "group": group_idx})

            all_data.append(group_data)

        return pd.concat(all_data, ignore_index=True)

    def get_generator_name(self) -> str:
        return "Crossing Curves KM"
