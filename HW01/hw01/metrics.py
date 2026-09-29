"""Metrics and cross-seed aggregation.

Reporting rules taken from the assignment:
  * every number is a mean +/- std over the 3 sampling configurations;
  * "significantly better" is banned -- comparisons are made by checking
    whether the intervals overlap;
  * Macro-F1 is the primary metric, worst-class recall the key safety metric.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 20) -> dict:
    """All first-submission metrics for a single run.

    A class absent from the validation split would make its recall undefined, so
    it is reported as NaN and excluded from the macro/worst-class averages
    rather than being silently counted as zero.
    """
    per_class_recall = recall_score(
        y_true, y_pred, labels=np.arange(num_classes), average=None, zero_division=0
    )
    present = np.unique(y_true)
    present = present[present < num_classes]

    worst_idx = int(present[np.argmin(per_class_recall[present])]) if present.size else -1
    return {
        "top1_accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=present, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "per_class_recall": per_class_recall.tolist(),
        "present_classes": present.tolist(),
        "worst_class_recall": float(per_class_recall[worst_idx]) if worst_idx >= 0 else float("nan"),
        "worst_class_index": worst_idx,
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=np.arange(num_classes)
        ).tolist(),
        "n_samples": int(y_true.size),
    }


def aggregate(runs: list[dict]) -> dict:
    """Collapse per-seed runs into mean +/- std blocks.

    ``per_class_recall`` and ``worst_class_index`` are averaged per seed
    *before* the worst class is selected, so the reported worst class is the
    one that is weak on average rather than the single worst draw.
    """
    keys = ("top1_accuracy", "macro_f1", "balanced_accuracy", "worst_class_recall")
    out: dict = {f"{k}_mean": float(np.mean([r[k] for r in runs])) for k in keys}
    out.update({f"{k}_std": float(np.std([r[k] for r in runs], ddof=1)) for k in keys})

    pcr = np.asarray([r["per_class_recall"] for r in runs], dtype=float)  # (n_seeds, C)
    present = np.asarray([r["present_classes"] for r in runs])[0]
    pcr_mean = np.nanmean(pcr, axis=0)
    pcr_std = np.nanstd(pcr, axis=0, ddof=1)

    out["per_class_recall_mean"] = pcr_mean.tolist()
    out["per_class_recall_std"] = pcr_std.tolist()
    worst = int(present[int(np.nanargmin(pcr_mean[present]))])
    out["worst_class_index"] = worst
    out["worst_class_recall_mean"] = float(pcr_mean[worst])
    out["worst_class_recall_std"] = float(pcr_std[worst])

    cm = np.mean([np.asarray(r["confusion_matrix"], dtype=float) for r in runs], axis=0)
    cm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1e-9)  # row-normalise
    out["mean_normalised_confusion_matrix"] = cm.tolist()
    out["seeds"] = [r.get("seed") for r in runs]
    return out


def format_pm(mean: float, std: float, digits: int = 4) -> str:
    """`0.8123 ± 0.0141` in the form the assignment requires."""
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def intervals_overlap(a_mean, a_std, b_mean, b_std, n_sd: float = 1.0) -> bool:
    """Do two mean +/- sd intervals overlap?

    Used to phrase every comparison. A non-overlap is reported as "the
    intervals are disjoint", never as "significantly better" -- three seeds
    cannot support a significance claim.
    """
    lo_a, hi_a = a_mean - n_sd * a_std, a_mean + n_sd * a_std
    lo_b, hi_b = b_mean - n_sd * b_std, b_mean + n_sd * b_std
    return not (hi_a < lo_b or hi_b < lo_a)


def describe_comparison(a_label, a, b_label, b) -> str:
    """One honest sentence comparing two aggregated metric blocks."""
    ov = intervals_overlap(a["macro_f1_mean"], a["macro_f1_std"], b["macro_f1_mean"], b["macro_f1_std"])
    delta = b["macro_f1_mean"] - a["macro_f1_mean"]
    worst_delta = b["worst_class_recall_mean"] - a["worst_class_recall_mean"]
    verdict = (
        "intervals overlap, so the difference is not separable with 3 seeds"
        if ov
        else "intervals are disjoint"
    )
    return (
        f"{b_label} vs {a_label}: Macro-F1 {format_pm(b['macro_f1_mean'], b['macro_f1_std'])} vs "
        f"{format_pm(a['macro_f1_mean'], a['macro_f1_std'])} (Δ={delta:+.4f}, {verdict}); "
        f"worst-class recall Δ={worst_delta:+.4f}"
    )


def rank_weak_classes(agg: dict, class_names: list[str], top_k: int = 5) -> list[tuple[str, int, float, float]]:
    """The ``top_k`` classes with the lowest mean recall across seeds.

    These are the targets for targeted augmentation in M1.
    """
    pcr_mean = np.asarray(agg["per_class_recall_mean"])
    pcr_std = np.asarray(agg["per_class_recall_std"])
    present = np.asarray([i for i, v in enumerate(pcr_mean) if not np.isnan(v)])
    order = present[np.argsort(pcr_mean[present])][:top_k]
    return [(class_names[i], int(i), float(pcr_mean[i]), float(pcr_std[i])) for i in order]
