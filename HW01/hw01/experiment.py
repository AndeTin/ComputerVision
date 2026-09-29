"""Shared training/evaluation engine.

One code path produces every number in the study, so M0/M1/M2/M3 remain
comparable. A "run" is (seed, geometry config, augmentation config) -> metrics.
The three seeds are always looped over; nothing is reported from a single run.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch
from torch.utils.data import DataLoader

from .augmentation import AugmentSpec, AugmentedFewShotDataset, expand_records
from .data import (
    IMAGENET_MEAN, IMAGENET_STD, SEEDS, build_transform, cache_key,
    compute_pixel_stats, make_loader, make_split, set_global_seed,
)
from .metrics import aggregate, compute_metrics
from .models import LogisticRegressionProbe, extract_features, get_feature_extractor


@dataclass
class RunConfig:
    """Everything that determines one trained probe."""

    label: str
    strategy: str = "resize_crop"
    interpolation: str = "bilinear"
    spec: AugmentSpec | None = None            # None => M0, no augmentation
    target_labels: set[int] = field(default_factory=set)
    n_replicates: int = 0
    C: float = 1.0
    use_custom_stats: bool = False

    def key(self) -> str:
        return cache_key(
            label=self.label, strategy=self.strategy, interpolation=self.interpolation,
            spec=self.spec.name if self.spec else None,
            strength=self.spec.strength if self.spec else None,
            cutout=self.spec.cutout_frac if self.spec else None,
            sharpen=self.spec.sharpen if self.spec else None,
            targets=sorted(self.target_labels), reps=self.n_replicates, C=self.C,
            custom_stats=self.use_custom_stats,
        )


def _group_accuracy(y_true: np.ndarray, y_pred: np.ndarray, groups: np.ndarray) -> dict:
    """Accuracy and macro-F1 split by the high/low resolution group.

    Reported because Q2 asks whether the strategies differ specifically on the
    low-resolution minority. A group with a handful of samples is flagged so
    its error bars are not over-read.
    """
    out = {}
    for g in ("high", "low"):
        m = groups == g
        n = int(m.sum())
        if n == 0:
            out[g] = {"n": 0, "accuracy": None, "macro_f1": None}
            continue
        yt, yp = y_true[m], y_pred[m]
        present = np.unique(yt)
        out[g] = {
            "n": n,
            "accuracy": float((yt == yp).mean()),
            "macro_f1": float(
                __import__("sklearn.metrics", fromlist=["f1_score"]).f1_score(
                    yt, yp, average="macro", labels=present, zero_division=0)
            ),
        }
    out["low_group_is_thin"] = out["low"]["n"] < 20
    return out


def run_config(cfg: RunConfig, seeds=SEEDS, extractor=None, verbose: bool = True,
               collect_pairs: bool = False):
    """Train and evaluate ``cfg`` under every seed; return aggregated metrics.

    ``collect_pairs`` additionally returns the first seed's feature matrices and
    record list, so semantic integrity can be measured from the same forward
    pass rather than a second extraction (Q3).
    """
    extractor = extractor or get_feature_extractor()
    per_seed: list[dict] = []
    pair_payload = None
    t0 = time.time()

    for seed in seeds:
        set_global_seed(seed)
        split = make_split(seed)

        mean, std = IMAGENET_MEAN, IMAGENET_STD
        if cfg.use_custom_stats:
            s = compute_pixel_stats(split.train, cfg.strategy, cfg.interpolation)
            mean, std = tuple(s["mean"]), tuple(s["std"])

        val_tf = build_transform(cfg.strategy, cfg.interpolation, mean, std)

        records3 = expand_records(split.train, cfg.target_labels, cfg.n_replicates)
        spec = cfg.spec or AugmentSpec("none")
        train_ds = AugmentedFewShotDataset(records3, val_tf, spec, cfg.target_labels, seed)
        gen = torch.Generator()
        gen.manual_seed(seed)
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, generator=gen)
        val_loader = make_loader(split.val, val_tf, batch_size=32, shuffle=False)

        Xtr, ytr, gtr, ptr = extract_features(
            extractor, train_loader, cache_key=f"{cfg.key()}_s{seed}_train")
        Xva, yva, gva, pva = extract_features(
            extractor, val_loader, cache_key=f"{cfg.key()}_s{seed}_val")

        probe = LogisticRegressionProbe(C=cfg.C, seed=seed).fit(Xtr, ytr)
        ypred = probe.predict(Xva)

        m = compute_metrics(yva, ypred, num_classes=20)
        m["seed"] = seed
        m["n_train_effective"] = int(len(ytr))
        m["n_train_originals"] = len(split.train)
        m["probe_n_iter"] = probe.n_iter
        m["by_resolution_group"] = _group_accuracy(yva, ypred, gva)
        per_seed.append(m)

        if collect_pairs and pair_payload is None:
            pair_payload = {
                "X_train": Xtr, "y_train": ytr, "paths_train": ptr,
                "X_val": Xva, "y_val": yva, "paths_val": pva,
                "records3": records3, "seed": seed,
            }
        if verbose:
            g = m["by_resolution_group"]
            print(f"    seed {seed:>4}  macro_f1 {m['macro_f1']:.4f}  acc {m['top1_accuracy']:.4f}  "
                  f"worst {m['worst_class_recall']:.3f}  "
                  f"acc[high] {g['high']['accuracy'] if g['high']['accuracy'] is not None else float('nan'):.3f}  "
                  f"acc[low] {g['low']['accuracy'] if g['low']['accuracy'] is not None else float('nan'):.3f}  "
                  f"n_train {len(ytr)}")

    agg = aggregate(per_seed)
    agg["config"] = {
        "label": cfg.label, "strategy": cfg.strategy, "interpolation": cfg.interpolation,
        "spec": cfg.spec.to_dict() if cfg.spec else None,
        "target_labels": sorted(cfg.target_labels), "n_replicates": cfg.n_replicates,
        "C": cfg.C, "use_custom_stats": cfg.use_custom_stats,
    }
    agg["per_seed"] = per_seed
    agg["by_resolution_group"] = _mean_group(per_seed)
    agg["wall_seconds"] = round(time.time() - t0, 1)
    return (agg, pair_payload) if collect_pairs else agg


def _mean_group(per_seed: list[dict]) -> dict:
    """Average the per-group accuracies across seeds into mean +/- sd."""
    out = {}
    for g in ("high", "low"):
        accs = [s["by_resolution_group"][g]["accuracy"] for s in per_seed
                if s["by_resolution_group"][g]["accuracy"] is not None]
        f1s = [s["by_resolution_group"][g]["macro_f1"] for s in per_seed
               if s["by_resolution_group"][g]["macro_f1"] is not None]
        ns = [s["by_resolution_group"][g]["n"] for s in per_seed]
        out[g] = {
            "n_mean": float(np.mean(ns)) if ns else 0.0,
            "accuracy_mean": float(np.mean(accs)) if accs else None,
            "accuracy_std": float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0,
            "macro_f1_mean": float(np.mean(f1s)) if f1s else None,
            "macro_f1_std": float(np.std(f1s, ddof=1)) if len(f1s) > 1 else 0.0,
        }
    return out
