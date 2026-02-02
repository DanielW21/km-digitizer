import io
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from base import (
    BaseKMGenerator,
    KMPlotConfig,
    apply_censoring,
    generate_flexible_weibull_survival_times,
)
from PIL import Image


class LowResolutionKMGenerator(BaseKMGenerator):
    """Generates low-resolution, pixelated KM plots"""

    def __init__(
        self,
        config: KMPlotConfig,
        target_dpi: int = 72,
        add_compression: bool = True,
        jpeg_quality: int = 60,
        random_seed: int = 42,
    ):
        super().__init__(config, random_seed)
        self.target_dpi = target_dpi
        self.add_compression = add_compression
        self.jpeg_quality = jpeg_quality
        self.config.dpi = target_dpi

    def generate_survival_data(self) -> pd.DataFrame:
        """Generate standard survival data"""
        survival_times = generate_flexible_weibull_survival_times(
            self.config.num_samples, 1.0, 10.0, self.random_seed
        )

        observed_times, events = apply_censoring(
            survival_times, self.config.censoring_rate, "random", self.random_seed
        )

        return pd.DataFrame({"time": observed_times, "event": events})

    def create_plot(self, data: pd.DataFrame, save_path: str = None) -> Tuple[plt.Figure, plt.Axes]:
        """Create low-resolution plot with compression artifacts"""
        fig, ax = super().create_plot(data, None)  # Don't save yet

        if save_path and self.add_compression:
            # Save as JPEG with low quality to introduce compression artifacts
            buffer = io.BytesIO()
            fig.savefig(buffer, format="png", dpi=self.target_dpi, bbox_inches="tight")
            buffer.seek(0)

            # Load and save the compressed image
            img = Image.open(buffer)

            # Convert RGBA to RGB if needed (JPEG doesn't support transparency)
            if img.mode in ("RGBA", "LA"):
                # Create white background
                background = Image.new("RGB", img.size, (255, 255, 255))
                if img.mode == "RGBA":
                    background.paste(img, mask=img.split()[-1])  # Use alpha channel as mask
                else:
                    background.paste(img, mask=img.split()[-1])
                img = background
            elif img.mode != "RGB":
                img = img.convert("RGB")

            img.save(save_path.replace(".png", ".jpg"), "JPEG", quality=self.jpeg_quality)

            # Also save as PNG for comparison
            img_png = Image.open(buffer)
            buffer.seek(0)  # Reset buffer
            img_png.save(save_path, "PNG")
        elif save_path:
            fig.savefig(save_path, dpi=self.target_dpi, bbox_inches="tight")

        return fig, ax

    def get_generator_name(self) -> str:
        return f"Low Resolution KM ({self.target_dpi} DPI)"


class NoisyKMGenerator(BaseKMGenerator):
    """Generates KM plots with various types of noise and artifacts"""

    def __init__(
        self,
        config: KMPlotConfig,
        noise_type: str = "gaussian",
        noise_intensity: float = 0.1,
        add_artifacts: bool = True,
        random_seed: int = 42,
    ):
        super().__init__(config, random_seed)
        self.noise_type = noise_type
        self.noise_intensity = noise_intensity
        self.add_artifacts = add_artifacts

    def generate_survival_data(self) -> pd.DataFrame:
        """Generate survival data"""
        survival_times = generate_flexible_weibull_survival_times(
            self.config.num_samples, 1.0, 10.0, self.random_seed
        )

        observed_times, events = apply_censoring(
            survival_times, self.config.censoring_rate, "random", self.random_seed
        )

        return pd.DataFrame({"time": observed_times, "event": events})

    def apply_visual_effects(self, fig, ax):
        """Apply noise and visual artifacts"""
        super().apply_visual_effects(fig, ax)

        if self.add_artifacts:
            n_artifacts = np.random.poisson(3)
            for _ in range(n_artifacts):
                x_pos = np.random.uniform(0.1, 0.9)
                y_pos = np.random.uniform(0.1, 0.9)
                artifact_text = np.random.choice(["●", "◆", "▲", "■", "×"])
                ax.text(
                    x_pos,
                    y_pos,
                    artifact_text,
                    transform=ax.transAxes,
                    fontsize=np.random.uniform(8, 16),
                    alpha=np.random.uniform(0.3, 0.7),
                    color=np.random.choice(["red", "blue", "green", "black"]),
                )

    def create_plot(self, data: pd.DataFrame, save_path: str = None) -> Tuple[plt.Figure, plt.Axes]:
        """Create noisy plot"""
        fig, ax = super().create_plot(data, None)

        if save_path:
            # Save to buffer first
            buffer = io.BytesIO()
            fig.savefig(buffer, format="png", dpi=self.config.dpi, bbox_inches="tight")
            buffer.seek(0)

            # Load image and add noise
            img = np.array(Image.open(buffer))

            if self.noise_type == "gaussian":
                noise = np.random.normal(0, self.noise_intensity * 255, img.shape).astype(np.int16)
                noisy_img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            elif self.noise_type == "salt_pepper":
                noisy_img = img.copy()
                # Salt noise
                salt_coords = tuple(
                    [
                        np.random.randint(0, i - 1, int(self.noise_intensity * img.size * 0.5))
                        for i in img.shape[:2]
                    ]
                )
                noisy_img[salt_coords] = 255
                # Pepper noise
                pepper_coords = tuple(
                    [
                        np.random.randint(0, i - 1, int(self.noise_intensity * img.size * 0.5))
                        for i in img.shape[:2]
                    ]
                )
                noisy_img[pepper_coords] = 0
            else:
                noisy_img = img

            # Save noisy image
            Image.fromarray(noisy_img).save(save_path)

        return fig, ax

    def get_generator_name(self) -> str:
        return f"Noisy KM ({self.noise_type}, {self.noise_intensity:.1%})"


class TruncatedAxisKMGenerator(BaseKMGenerator):
    """Generates KM plots with truncated Y-axis (not starting from 0)"""

    def __init__(
        self, config: KMPlotConfig, y_min: float = 0.4, y_max: float = 1.0, random_seed: int = 42
    ):
        super().__init__(config, random_seed)
        # Override y-axis limits
        self.config.y_limits = (y_min, y_max)
        self.config.truncate_y_axis = True
        self.config.y_truncate_min = y_min
        self.config.y_truncate_max = y_max

    def generate_survival_data(self) -> pd.DataFrame:
        """Generate survival data that will be displayed in truncated range"""
        # Generate data that mostly falls in the truncated range
        survival_times = np.random.gamma(
            2, 5, size=self.config.num_samples
        )  # Gamma for more variation

        observed_times, events = apply_censoring(
            survival_times, self.config.censoring_rate, "random", self.random_seed
        )

        return pd.DataFrame({"time": observed_times, "event": events})

    def get_generator_name(self) -> str:
        return f"Truncated Y-Axis KM ({self.config.y_truncate_min:.1f}-{self.config.y_truncate_max:.1f})"


class OverlappingCurvesKMGenerator(BaseKMGenerator):
    """Generates KM plots with very closely overlapping survival curves"""

    def __init__(self, config: KMPlotConfig, separation_factor: float = 0.1, random_seed: int = 42):
        super().__init__(config, random_seed)
        self.separation_factor = separation_factor

        if config.num_groups < 2:
            # Force multiple groups for overlapping
            self.config.num_groups = 3

    def generate_survival_data(self) -> pd.DataFrame:
        """Generate survival data with very similar group outcomes"""
        all_data = []
        base_scale = 10.0
        samples_per_group = self.config.num_samples // self.config.num_groups

        for group_idx in range(self.config.num_groups):
            # Very small differences between groups
            group_scale = base_scale * (1 + (group_idx - 1) * self.separation_factor)
            group_shape = 1.0 + group_idx * 0.1  # Slight shape variation

            survival_times = generate_flexible_weibull_survival_times(
                samples_per_group, group_shape, group_scale, self.random_seed + group_idx
            )

            observed_times, events = apply_censoring(
                survival_times, self.config.censoring_rate, "random", self.random_seed + group_idx
            )

            group_data = pd.DataFrame({"time": observed_times, "event": events, "group": group_idx})

            all_data.append(group_data)

        return pd.concat(all_data, ignore_index=True)

    def get_generator_name(self) -> str:
        return f"Overlapping Curves KM (sep={self.separation_factor:.1f})"


class SmallSampleKMGenerator(BaseKMGenerator):
    """Generates KM plots with very small sample sizes (high uncertainty)"""

    def __init__(self, config: KMPlotConfig, random_seed: int = 42):
        super().__init__(config, random_seed)
        # Override to use small sample size
        self.config.num_samples = np.random.randint(10, 25)  # Very small samples
        self.config.show_confidence_intervals = True  # Show uncertainty
        self.config.ci_alpha = 0.5  # More visible CI

    def generate_survival_data(self) -> pd.DataFrame:
        """Generate survival data with small sample"""
        survival_times = generate_flexible_weibull_survival_times(
            self.config.num_samples, 1.0, 10.0, self.random_seed
        )

        # Higher censoring rate for small samples makes it more challenging
        observed_times, events = apply_censoring(
            survival_times, min(0.5, self.config.censoring_rate * 1.5), "random", self.random_seed
        )

        return pd.DataFrame({"time": observed_times, "event": events})

    def get_generator_name(self) -> str:
        return f"Small Sample KM (n={self.config.num_samples})"
