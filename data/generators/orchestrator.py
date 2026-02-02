import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from base import BaseKMGenerator, KMPlotConfig
from edge_cases import (
    LowResolutionKMGenerator,
    NoisyKMGenerator,
    RealWorldComplexKMGenerator,
    TruncatedAxisKMGenerator,
)
from standard import (
    CrossingCurvesKMGenerator,
    EarlyDropoffKMGenerator,
    MultiGroupKMGenerator,
    PlateauKMGenerator,
    StandardKMGenerator,
)


class KMDatasetGenerator:
    """Main class for generating comprehensive KM plot datasets"""

    def __init__(self, output_dir: str, random_seed: int = 42):
        self.output_dir = Path(output_dir)
        self.random_seed = random_seed
        np.random.seed(random_seed)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "plots").mkdir(exist_ok=True)
        (self.output_dir / "data").mkdir(exist_ok=True)
        (self.output_dir / "metadata").mkdir(exist_ok=True)

        self.generators_created = 0

    def create_generator_set(
        self, include_types: List[str] = None, samples_per_type: int = 5
    ) -> List[BaseKMGenerator]:
        """
        Create a set of generators grouped by type
        """

        if include_types is None:
            include_types = [
                "standard",
                "multi_group",
                "special_patterns",
                "low_resolution",
                "noisy_images",
                "truncated_axis",
                "real_world_complex",
            ]

        generators = []

        # GROUP 1: Standard single-group plots
        if "standard" in include_types:
            print(f"Creating {samples_per_type} standard KM plots...")
            for i in range(samples_per_type):
                config = KMPlotConfig(
                    num_samples=np.random.randint(50, 200),
                    censoring_rate=np.random.uniform(0.1, 0.4),
                    figure_size=(10, 6),
                    dpi=150,
                    show_grid=True,
                )

                shape = np.random.choice([0.5, 1.0, 1.5, 2.0])  # Different Weibull shapes
                scale = np.random.uniform(8.0, 12.0)
                gen = StandardKMGenerator(
                    config,
                    shape=shape,
                    scale=scale,
                    random_seed=self.random_seed + self.generators_created,
                )
                generators.append(gen)
                self.generators_created += 1

        # GROUP 2: Multi-group comparison plots
        if "multi_group" in include_types:
            print(f"Creating {samples_per_type} multi-group KM plots...")
            for i in range(samples_per_type):
                n_groups = np.random.randint(2, 4)
                config = KMPlotConfig(
                    num_samples=np.random.randint(100, 300),
                    censoring_rate=np.random.uniform(0.2, 0.5),
                    num_groups=n_groups,
                    group_names=[f"Treatment {j+1}" for j in range(n_groups)],
                    show_legend=True,
                    show_confidence_intervals=np.random.choice([True, False]),
                )

                group_effects = np.random.uniform(0.5, 2.0, n_groups)
                gen = MultiGroupKMGenerator(
                    config,
                    group_effects=group_effects.tolist(),
                    random_seed=self.random_seed + self.generators_created,
                )
                generators.append(gen)
                self.generators_created += 1

        # GROUP 3: Special pattern plots (early dropoff, plateau, crossing)
        if "special_patterns" in include_types:
            pattern_generators = [
                (EarlyDropoffKMGenerator, "early dropoff"),
                (PlateauKMGenerator, "plateau"),
                (CrossingCurvesKMGenerator, "crossing curves"),
            ]

            for GeneratorClass, name in pattern_generators:
                print(f"Creating {samples_per_type//3} {name} KM plots...")
                for i in range(samples_per_type // 3):
                    if GeneratorClass == CrossingCurvesKMGenerator:
                        config = KMPlotConfig(
                            num_samples=np.random.randint(100, 200), num_groups=2, show_legend=True
                        )
                    else:
                        config = KMPlotConfig(num_samples=np.random.randint(80, 180))

                    if GeneratorClass == PlateauKMGenerator:
                        gen = GeneratorClass(
                            config,
                            cure_fraction=np.random.uniform(0.1, 0.3),
                            random_seed=self.random_seed + self.generators_created,
                        )
                    else:
                        gen = GeneratorClass(
                            config, random_seed=self.random_seed + self.generators_created
                        )

                    generators.append(gen)
                    self.generators_created += 1

        # GROUP 4: Low resolution plots
        if "low_resolution" in include_types:
            print(f"Creating {samples_per_type} low-resolution KM plots...")
            for i in range(samples_per_type):
                config = KMPlotConfig(num_samples=np.random.randint(50, 150))
                gen = LowResolutionKMGenerator(
                    config,
                    target_dpi=np.random.randint(72, 100),
                    jpeg_quality=np.random.randint(50, 80),
                    random_seed=self.random_seed + self.generators_created,
                )
                generators.append(gen)
                self.generators_created += 1

        # GROUP 5: Noisy image plots
        if "noisy_images" in include_types:
            print(f"Creating {samples_per_type} noisy KM plots...")
            for i in range(samples_per_type):
                config = KMPlotConfig(num_samples=np.random.randint(80, 180))
                noise_type = np.random.choice(["gaussian", "salt_pepper"])
                gen = NoisyKMGenerator(
                    config,
                    noise_type=noise_type,
                    noise_intensity=np.random.uniform(0.05, 0.2),
                    random_seed=self.random_seed + self.generators_created,
                )
                generators.append(gen)
                self.generators_created += 1

        # GROUP 6: Truncated axis plots
        if "truncated_axis" in include_types:
            print(f"Creating {samples_per_type} truncated-axis KM plots...")
            for i in range(samples_per_type):
                config = KMPlotConfig(num_samples=np.random.randint(60, 160))
                y_min = np.random.uniform(0.3, 0.6)
                y_max = np.random.uniform(0.85, 1.0)
                gen = TruncatedAxisKMGenerator(
                    config,
                    y_min=y_min,
                    y_max=y_max,
                    random_seed=self.random_seed + self.generators_created,
                )
                generators.append(gen)
                self.generators_created += 1

        # GROUP 7: Complex real-world scenarios
        if "real_world_complex" in include_types:
            print(f"Creating {samples_per_type} complex real-world KM plots...")
            for i in range(samples_per_type):
                config = KMPlotConfig(
                    num_samples=np.random.randint(100, 400), show_confidence_intervals=True
                )

                complexity = np.random.randint(3, 6)
                gen = RealWorldComplexKMGenerator(
                    config,
                    complexity_level=complexity,
                    random_seed=self.random_seed + self.generators_created,
                )
                generators.append(gen)
                self.generators_created += 1

        return generators

    def generate_single(self, generator: BaseKMGenerator, plot_id: str) -> Dict[str, Any]:
        """Generate a single plot and return metadata"""
        try:
            filename = f"{plot_id}_{generator.get_generator_name().lower().replace(' ', '_').replace('(', '').replace(')', '').replace('.', '')}"
            metadata = generator.generate_and_save(
                str(self.output_dir),
                filename,
                plots_dir=str(self.output_dir / "plots"),
                data_dir=str(self.output_dir / "data"),
                metadata_dir=str(self.output_dir / "metadata"),
            )

            metadata["plot_id"] = plot_id
            metadata["generator_class"] = generator.__class__.__name__
            metadata["random_seed"] = generator.random_seed
            metadata["generation_timestamp"] = np.datetime64("now").astype(str)

            return metadata

        except Exception as e:
            print(f"Error generating plot {plot_id}: {str(e)}")
            return {
                "plot_id": plot_id,
                "error": str(e),
                "generator": generator.get_generator_name(),
            }

    def generate_dataset(
        self, include_types: List[str] = None, samples_per_type: int = 5, save_summary: bool = True
    ) -> Dict[str, Any]:
        """Generate a complete dataset of KM plots"""

        print(f"Creating generators...")
        generators = self.create_generator_set(include_types, samples_per_type)
        print(f"Created {len(generators)} generators")

        all_metadata = []
        for i, generator in enumerate(generators):
            print(f"Generating plot {i+1}/{len(generators)}: {generator.get_generator_name()}")
            metadata = self.generate_single(generator, f"plot_{i:04d}")
            all_metadata.append(metadata)

        dataset_summary = {
            "total_plots": len(all_metadata),
            "include_types": include_types,
            "samples_per_type": samples_per_type,
            "generation_seed": self.random_seed,
            "successful_generations": len([m for m in all_metadata if "error" not in m]),
            "failed_generations": len([m for m in all_metadata if "error" in m]),
            "generator_types": {},
            "plots": all_metadata,
        }

        # Count generator types
        for metadata in all_metadata:
            if "generator" in metadata:
                gen_name = metadata["generator"]
                if gen_name not in dataset_summary["generator_types"]:
                    dataset_summary["generator_types"][gen_name] = 0
                dataset_summary["generator_types"][gen_name] += 1

        if save_summary:
            summary_path = self.output_dir / "dataset_summary.json"
            with open(summary_path, "w") as f:
                json.dump(dataset_summary, f, indent=2, default=str)
            print(f"Dataset summary saved to: {summary_path}")

        print(
            f"Generation complete! {dataset_summary['successful_generations']} plots generated successfully"
        )
        if dataset_summary["failed_generations"] > 0:
            print(f"Warning: {dataset_summary['failed_generations']} plots failed to generate")

        return dataset_summary


def create_benchmark_dataset(
    output_dir: str, size: int, include_types: List[str] = None, random_seed: int = 42
) -> Dict[str, Any]:
    """Create a standardized benchmark dataset

    Args:
        output_dir: Directory to save the dataset
        size: Number of plots per type to generate
        include_types: List of plot types to include (None = all types)
        random_seed: Random seed for reproducibility
    """

    generator = KMDatasetGenerator(output_dir, random_seed)

    return generator.generate_dataset(
        include_types=include_types, samples_per_type=size, save_summary=True
    )


if __name__ == "__main__":
    output_dir = "../synthetic_data/benchmark"

    print("Generating benchmark dataset...")
    summary = create_benchmark_dataset(output_dir, size=5, random_seed=42)

    print("\nDataset Summary:")
    print(f"Total plots: {summary['total_plots']}")
    print(f"Successful: {summary['successful_generations']}")
