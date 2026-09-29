"""Report-facing views over the full result payloads.

The stage scripts write verbose JSON to ``results/full/`` so that any figure can
be redrawn without rerunning training. That verbosity is a liability when reading
the results, so :func:`build_summary` assembles a single
``results/summary.json`` holding the headline table for M0 and M1, the Q2
strategy ranking, the Q3 ratio sweep, and ready-to-paste markdown.

Two files, two audiences:

* ``results/summary.json``  -- what the write-up quotes. Small, stable schema.
* ``results/full/*.json``   -- the archive. Verbose, per-seed, regenerates every
  figure.
"""

from __future__ import annotations

import datetime as _dt
import json
import os

import numpy as np

from .data import SEEDS, SUMMARY_PATH, save_json
from .metrics import format_pm, intervals_overlap


def _pm(agg: dict, key: str = "macro_f1") -> str:
    return format_pm(agg[f"{key}_mean"], agg[f"{key}_std"])


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    """A GitHub-flavoured markdown table, ready to paste into the write-up.

    Cell values are escaped: geometry labels contain a literal ``|``
    ("direct|bilinear"), which would otherwise be read as a column break and
    silently shift every cell to its left.
    """
    def cell(c: object) -> str:
        return str(c).replace("|", "\\|")

    lines = ["| " + " | ".join(cell(h) for h in headers) + " |",
             "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        if len(r) != len(headers):
            raise ValueError(
                f"row has {len(r)} cells but header has {len(headers)}: {r!r}")
        lines.append("| " + " | ".join(cell(c) for c in r) + " |")
    return "\n".join(lines)


def build_summary(q1: dict | None, q2: dict | None, q3: dict | None) -> dict:
    """Assemble ``results/summary.json`` from whichever stages were run.

    Each argument may be ``None`` if that stage was skipped, so ``--stage q3``
    still produces a valid (if partial) summary.

    Sections for stages that were *not* re-run are carried over from the
    existing ``summary.json`` on disk, so running one stage incrementally does
    not discard the other stages' results. Pass an explicit stage to refresh
    just that section; run with no arguments for a fully self-consistent file.
    """
    class_names = [f"cl{i:02d}" for i in range(1, 21)]

    previous: dict = {}
    if os.path.exists(SUMMARY_PATH):
        try:
            with open(SUMMARY_PATH) as fh:
                loaded = json.load(fh)
            if isinstance(loaded, dict):
                previous = loaded
        except (json.JSONDecodeError, OSError):
            previous = {}  # corrupt file: rebuild from scratch

    summary: dict = {
        "generated": _dt.datetime.now().isoformat(timespec="seconds"),
        "seeds": list(SEEDS),
        "reporting_rule": (
            "Every number is a mean +/- standard deviation over the 3 sampling "
            "configurations (seeds 42, 420, 4200). Comparisons are stated by "
            "whether the +/-1 sd intervals overlap; three seeds cannot support "
            "a significance claim."
        ),
    }
    # Carry forward any stage not re-run in this invocation.
    for key in ("headline", "per_class_recall", "q3_ratio_sweep", "q2", "q1"):
        if key in previous:
            summary[key] = previous[key]

    # ---- headline M0 vs M1
    if q3:
        m0, m1 = q3["m0"], q3["best_m1"]
        overlap = intervals_overlap(
            m0["macro_f1_mean"], m0["macro_f1_std"],
            m1["macro_f1_mean"], m1["macro_f1_std"],
        )
        summary["headline"] = {
            "M0": {
                "top1": _pm(m0, "top1_accuracy"),
                "macro_f1": _pm(m0),
                "worst_class_recall": _pm(m0, "worst_class_recall"),
                "worst_class": class_names[m0["worst_class_index"]],
            },
            "M1": {
                "recipe": q3["best_recipe"],
                "ratio": q3["turning_points"][q3["best_recipe"]]["best_ratio"],
                "top1": _pm(m1, "top1_accuracy"),
                "macro_f1": _pm(m1),
                "worst_class_recall": _pm(m1, "worst_class_recall"),
            },
            "m1_minus_m0_macro_f1": round(m1["macro_f1_mean"] - m0["macro_f1_mean"], 4),
            "m1_minus_m0_worst_recall": round(
                m1["worst_class_recall_mean"] - m0["worst_class_recall_mean"], 4),
            "intervals_overlap": overlap,
            "verdict": (
                "M1 does not separate from M0: the +/-1 sd intervals overlap, so "
                "targeted augmentation is NOT shown to help at this sample size."
                if overlap else
                "M1 separates from M0: the +/-1 sd intervals are disjoint."
            ),
            "best_ratio_turning_point": q3["turning_points"][q3["best_recipe"]]["turning_point"],
        }
        summary["headline"]["table"] = _md_table(
            ["model", "Top-1", "Macro-F1", "worst-class recall"],
            [["M0", summary["headline"]["M0"]["top1"],
              summary["headline"]["M0"]["macro_f1"],
              summary["headline"]["M0"]["worst_class_recall"]],
             ["M1", summary["headline"]["M1"]["top1"],
              summary["headline"]["M1"]["macro_f1"],
              summary["headline"]["M1"]["worst_class_recall"]]],
        )

        # ---- per-class recall table
        p0 = np.asarray(m0["per_class_recall_mean"])
        s0 = np.asarray(m0["per_class_recall_std"])
        p1 = np.asarray(m1["per_class_recall_mean"])
        targeted = set(q3["target_labels"])
        summary["per_class_recall"] = {
            "table": _md_table(
                ["class", "M0", "sd", "M1", "Δ", "targeted"],
                [[class_names[i], f"{p0[i]:.3f}", f"{s0[i]:.3f}", f"{p1[i]:.3f}",
                  f"{p1[i]-p0[i]:+.3f}", "yes" if i in targeted else ""]
                 for i in range(20)],
            ),
            "values": {class_names[i]: {"m0": float(p0[i]), "m1": float(p1[i]),
                                       "targeted": i in targeted} for i in range(20)},
        }

        # ---- Q3 ratio sweep
        summary["q3_ratio_sweep"] = {
            "weak_classes_targeted": [w["class"] for w in q3["weak_classes"]],
            "semantic_integrity": {
                name: {"cosine_to_source_mean": r["cosine_to_source_mean"],
                       "prototype_agreement": r["prototype_agreement"],
                       "verdict": r["interpretation"]}
                for name, r in q3["semantic_integrity"].items()
            },
            "turning_points": {
                name: {"best_ratio": p["best_ratio"],
                       "best_macro_f1": round(p["best_macro_f1"], 4),
                       "turning_point": (p["turning_point"] or {}).get("ratio"),
                       "verdict": p["verdict"]}
                for name, p in q3["turning_points"].items()
            },
            "failure_cases_unrecovered": q3["failure_cases_unrecovered"],
            "table": _md_table(
                ["recipe", "ratio", "Macro-F1", "worst-class recall", "turning point"],
                [[name, f"{r:g}x", f"{p['macro_f1'][i]:.4f} ± {p['macro_f1_std'][i]:.4f}",
                  f"{p['worst_class_recall'][i]:.3f}",
                  f"{p['turning_point']['ratio']:g}x" if p["turning_point"] else "not established"]
                 for name, p in q3["turning_points"].items()
                 for i, r in enumerate(p["ratios"])],
            ),
        }

    # ---- Q2
    if q2:
        pa = q2["padding_audit"]
        summary["q2"] = {
            "best_geometry": q2["best_config"],
            "worst_geometry": q2["worst_config"],
            "strategy_ranking": q2["strategy_ranking"],
            "table": _md_table(
                ["geometry", "Macro-F1", "Top-1", "worst-class recall",
                 "% w/ zero fill", "% content cropped"],
                [[k, _pm(v), _pm(v, "top1_accuracy"), _pm(v, "worst_class_recall"),
                  f"{pa[k.split('|')[0]]['pct_with_synthetic_fill']:.1f}%",
                  f"{pa[k.split('|')[0]]['pct_with_discarded_content']:.1f}%"]
                 for k, v in q2["results"].items()],
            ),
            "padding_audit": pa,
            "comparisons": q2["comparisons"],
        }

    # ---- Q1
    if q1:
        summary["q1"] = {
            "n_images": q1["dataset_profile"]["n_images"],
            "n_low_res": q1["dataset_profile"]["n_low_res"],
            "aspect_gt_1p14": q1["dataset_profile"]["aspect_ratio"]["n_gt_1p14"],
            "split_rule": q1["split_report"]["rule"],
            "exif": {
                "n_with_exif_tag": q1["exif_audit"]["n_with_exif_tag"],
                "n_changed": q1["exif_audit"]["n_geometry_changed_by_exif_transpose"],
                "conclusion": q1["exif_audit"]["conclusion"],
            },
            "normalisation": {
                "imagenet_mean": q1["normalisation"]["imagenet_mean"],
                "dataset_mean_resize_crop":
                    q1["normalisation"]["per_strategy"]["resize_crop"]["mean"],
                "max_mean_abs_delta":
                    q1["normalisation"]["per_strategy"]["resize_crop"]["max_mean_abs_delta"],
                "verdict": q1["normalisation"]["verdict"],
            },
        }

    save_json(summary, "summary.json", full=False)
    return summary
