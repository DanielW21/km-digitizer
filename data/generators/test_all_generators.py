import os

import matplotlib
from base import KMPlotConfig
from edge_cases import (
    LowResolutionKMGenerator,
    NoisyKMGenerator,
    OverlappingCurvesKMGenerator,
    TruncatedAxisKMGenerator,
)
from standard import (
    CrossingCurvesKMGenerator,
    EarlyDropoffKMGenerator,
    MultiGroupKMGenerator,
    PlateauKMGenerator,
    StandardKMGenerator,
)


def test_all_generators(test_cases):
    matplotlib.use("Agg")
    output_dir = "../synthetic_data/comprehensive_test"
    os.makedirs(output_dir, exist_ok=True)

    test_results = []
    print(f"✓ Starting tests... Output: {output_dir}\n")

    for gen_class, suffix, params in test_cases:
        test_id = f"{gen_class.__name__}-{suffix}"
        try:
            # Set default config; adjust for specific needs if necessary
            num_groups = 2 if "Crossing" in test_id or "Multi" in test_id else 1
            if "Overlapping" in test_id:
                num_groups = 3

            config = KMPlotConfig(num_samples=50, num_groups=num_groups, show_legend=True)

            # Initialize and Run
            gen = gen_class(config, **params)
            gen.generate_and_save(output_dir, f"test_{suffix}")

            print(f"✓ Passed: {test_id}")
            test_results.append((test_id, True, None))
        except Exception as e:
            print(f"✗ Failed: {test_id} - {str(e)}")
            test_results.append((test_id, False, str(e)))

    # Summary
    passed = sum(1 for _, success, _ in test_results if success)
    print(f"\n{'='*30}\nTEST SUMMARY: {passed}/{len(test_results)} Passed\n{'='*30}")

    for name, success, error in test_results:
        if not success:
            print(f"ERROR in {name}: {error}")


if __name__ == "__main__":
    test_cases = [
        (StandardKMGenerator, "standard_exp", {"shape": 1.0, "scale": 10.0}),
        (StandardKMGenerator, "standard_early", {"shape": 0.5, "scale": 8.0}),
        (StandardKMGenerator, "standard_late", {"shape": 2.0, "scale": 12.0}),
        (MultiGroupKMGenerator, "multigroup", {"group_effects": [1.0, 1.5], "shapes": [0.8, 1.5]}),
        (EarlyDropoffKMGenerator, "early_dropoff", {"shape": 0.4, "scale": 6.0}),
        (PlateauKMGenerator, "plateau", {"cure_fraction": 0.2, "susceptible_shape": 1.3}),
        (
            CrossingCurvesKMGenerator,
            "crossing",
            {"early_shapes": [1.5, 0.7], "late_shapes": [0.8, 1.8]},
        ),
        (LowResolutionKMGenerator, "lowres", {"target_dpi": 72, "jpeg_quality": 60}),
        (NoisyKMGenerator, "noisy", {"noise_type": "gaussian", "noise_intensity": 0.1}),
        (TruncatedAxisKMGenerator, "truncated", {"y_min": 0.4, "y_max": 1.0}),
        (OverlappingCurvesKMGenerator, "overlapping", {"separation_factor": 0.05}),
    ]

    test_all_generators(test_cases)
