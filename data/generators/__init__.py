from base import BaseKMGenerator, KMPlotConfig
from edge_cases import (
    LowResolutionKMGenerator,
    NoisyKMGenerator,
    OverlappingCurvesKMGenerator,
    SmallSampleKMGenerator,
    TruncatedAxisKMGenerator,
)
from orchestrator import KMDatasetGenerator, create_benchmark_dataset
from standard import (
    CrossingCurvesKMGenerator,
    EarlyDropoffKMGenerator,
    MultiGroupKMGenerator,
    PlateauKMGenerator,
    StandardKMGenerator,
)

__version__ = "1.0.0"

__all__ = [
    # Base classes
    "BaseKMGenerator",
    "KMPlotConfig",
    # Standard generators
    "StandardKMGenerator",
    "MultiGroupKMGenerator",
    "EarlyDropoffKMGenerator",
    "PlateauKMGenerator",
    "CrossingCurvesKMGenerator",
    # Edge case generators
    "LowResolutionKMGenerator",
    "NoisyKMGenerator",
    "TruncatedAxisKMGenerator",
    "OverlappingCurvesKMGenerator",
    "SmallSampleKMGenerator",
    "RealWorldComplexKMGenerator",
    # Main orchestrator
    "KMDatasetGenerator",
    "create_benchmark_dataset",
]


def get_available_generators():
    """Return a list of all available generator classes"""
    return [
        StandardKMGenerator,
        MultiGroupKMGenerator,
        EarlyDropoffKMGenerator,
        PlateauKMGenerator,
        CrossingCurvesKMGenerator,
        LowResolutionKMGenerator,
        NoisyKMGenerator,
        TruncatedAxisKMGenerator,
        OverlappingCurvesKMGenerator,
        SmallSampleKMGenerator,
    ]
