"""HW01 library: data pipeline, model, metrics, augmentation, experiment engine.

Import surface for the stage scripts and for later submissions:

    from hw01.data import SEEDS, make_split, build_transform
    from hw01.models import get_feature_extractor, LogisticRegressionProbe
    from hw01.experiment import RunConfig, run_config

Nothing here imports from ``stages`` or ``run_experiments``, so the dependency
direction stays one-way: entry point -> stages -> library.
"""

from .augmentation import SPECS, AugmentSpec
from .data import (
    IMAGENET_MEAN, IMAGENET_STD, N_VAL_PER_CLASS, SEEDS,
    Split, build_transform, make_split,
)
from .experiment import RunConfig, run_config
from .metrics import compute_metrics, format_pm, intervals_overlap, rank_weak_classes
from .models import LogisticRegressionProbe, get_feature_extractor

__all__ = [
    "SEEDS", "N_VAL_PER_CLASS", "IMAGENET_MEAN", "IMAGENET_STD", "Split",
    "make_split", "build_transform",
    "get_feature_extractor", "LogisticRegressionProbe",
    "RunConfig", "run_config",
    "compute_metrics", "format_pm", "intervals_overlap", "rank_weak_classes",
    "AugmentSpec", "SPECS",
]
