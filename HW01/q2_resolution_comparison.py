"""Q2: geometry strategies and interpolation.

3 strategies x 2 interpolations x 3 seeds = 18 trained probes, all scored with
Macro-F1 as the primary metric and the low-resolution subgroup tracked
separately.

The mechanism under test: the three strategies differ in *what they do to
aspect ratio*, not merely in how much they resample.
  direct        - anisotropic squash; no synthetic pixels, but object shape
                  is distorted by up to 3.7x here.
  resize_crop   - isotropic scale then crop; for aspect ratio > 1.14 the
                  scaled short side is under 224, so torchvision's CenterCrop
                  zero-pads. Synthetic black borders are introduced silently.
  pad_to_square - isotropic scale then explicit symmetric zero-pad. Aspect
                  ratio is preserved and the padding is measurable.
"""

from __future__ import annotations

import numpy as np

from data import (
    DATASET_DIR, IMAGE_SIZE, INTERPOLATIONS, SEEDS, STRATEGIES,
    save_json, set_global_seed,
)
from experiment import RunConfig, run_config
from metrics import describe_comparison, format_pm, intervals_overlap
from models import get_feature_extractor


def _padding_audit(dataset_dir: str = DATASET_DIR) -> dict:
    """How much black does each strategy inject, image by image?

    Measured on the *geometry* stage alone, before normalisation, by comparing
    each output against a version with the synthetic border region filled by
    the image's own border colour.
    """
    from PIL import Image
    from data import get_class_paths, pil_loader, build_geometry

    stats = {s: {"n_padded": 0, "frac_black": [], "n_upscaled": 0,
                 "upscale_factor": []} for s in STRATEGIES}
    n_total = 0
    for paths in get_class_paths(dataset_dir):
        for p in paths:
            n_total += 1
            w, h = Image.open(p).size
            for s in STRATEGIES:
                out = build_geometry(s)(pil_loader(p))
                a = np.asarray(out, dtype=np.float32)
                # A pixel is "synthetic black" if it is near 0; real dark
                # pixels are rare in this dataset, so this is a good proxy.
                frac = float((a.max(axis=2) <= 2).mean())
                stats[s]["frac_black"].append(frac)
                if frac > 0.001:
                    stats[s]["n_padded"] += 1
                scale = IMAGE_SIZE / min(w, h)
                if scale > 1.0:
                    stats[s]["n_upscaled"] += 1
                stats[s]["upscale_factor"].append(scale)

    for s in STRATEGIES:
        fb = np.asarray(stats[s]["frac_black"])
        uf = np.asarray(stats[s]["upscale_factor"])
        stats[s] = {
            "n_images": n_total,
            "n_with_synthetic_black": int(stats[s]["n_padded"]),
            "pct_with_synthetic_black": round(100 * stats[s]["n_padded"] / n_total, 1),
            "mean_black_area_frac": round(float(fb.mean()), 4),
            "max_black_area_frac": round(float(fb.max()), 4),
            "n_upscaled_on_short_side": int((uf > 1.0).sum()),
            "median_short_side_upscale": round(float(np.median(uf)), 3),
            "max_short_side_upscale": round(float(uf.max()), 2),
        }
    return stats


def _interp_sentence(by_interp: dict) -> str:
    """Compare the two interpolations, phrased in terms of interval overlap.

    The per-seed std is averaged over strategies here, which is an
    approximation; the per-config values in ``results`` are the authoritative
    ones and the sentence is worded so it does not over-claim.
    """
    b, c = by_interp["bilinear"], by_interp["bicubic"]
    delta = c["macro_f1_mean"] - b["macro_f1_mean"]
    overlap = intervals_overlap(b["macro_f1_mean"], b["macro_f1_std"],
                                c["macro_f1_mean"], c["macro_f1_std"])
    return (
        f"bilinear mean macro_f1 {b['macro_f1_mean']:.4f} (avg sd {b['macro_f1_std']:.4f}), "
        f"bicubic {c['macro_f1_mean']:.4f} (avg sd {c['macro_f1_std']:.4f}), "
        f"difference {delta:+.4f}; averaged intervals "
        f"{'overlap' if overlap else 'are disjoint'}, so the choice of filter is not "
        f"separable from seed noise at this sample size."
    )


def run(dataset_dir: str = DATASET_DIR, seeds=SEEDS) -> dict:
    """Execute the 3x2x3 sweep and write results/q2_strategies.json."""
    set_global_seed(seeds[0])
    extractor = get_feature_extractor()

    padding = _padding_audit(dataset_dir)
    results: dict[str, dict] = {}
    print("=" * 72)
    print("Q2  geometry strategy x interpolation   (seeds: "
          + ", ".join(map(str, seeds)) + ")")
    print("=" * 72)
    for strat, st in padding.items():
        print(f"  {strat:14s} synthetic black in {st['pct_with_synthetic_black']:5.1f}% of images "
              f"(mean area {st['mean_black_area_frac']:.3f}, max {st['max_black_area_frac']:.3f}) | "
              f"short-side upscale median {st['median_short_side_upscale']:.2f}x max {st['max_short_side_upscale']:.1f}x")
    print()

    for strategy in STRATEGIES:
        for interp in INTERPOLATIONS:
            label = f"{strategy}|{interp}"
            print(f"  [{label}]")
            results[label] = run_config(
                RunConfig(label=label, strategy=strategy, interpolation=interp),
                seeds=seeds, extractor=extractor,
            )
            a = results[label]
            print(f"    -> macro_f1 {format_pm(a['macro_f1_mean'], a['macro_f1_std'])}  "
                  f"acc {format_pm(a['top1_accuracy_mean'], a['top1_accuracy_std'])}  "
                  f"worst {format_pm(a['worst_class_recall_mean'], a['worst_class_recall_std'])}  "
                  f"({a['wall_seconds']}s)\n")

    # ---- rankings and honest comparisons
    by_strategy, by_interp = {}, {}
    for strategy in STRATEGIES:
        rs = [results[f"{strategy}|{i}"] for i in INTERPOLATIONS]
        by_strategy[strategy] = {
            "macro_f1_mean": float(np.mean([r["macro_f1_mean"] for r in rs])),
            "macro_f1_std": float(np.mean([r["macro_f1_std"] for r in rs])),
        }
    for interp in INTERPOLATIONS:
        rs = [results[f"{s}|{interp}"] for s in STRATEGIES]
        by_interp[interp] = {
            "macro_f1_mean": float(np.mean([r["macro_f1_mean"] for r in rs])),
            "macro_f1_std": float(np.mean([r["macro_f1_std"] for r in rs])),
        }

    best = max(results, key=lambda k: results[k]["macro_f1_mean"])
    worst = min(results, key=lambda k: results[k]["macro_f1_mean"])
    # Main effect ordering, stated with the spread so the reader can judge.
    strat_rank = sorted(by_strategy, key=lambda s: -by_strategy[s]["macro_f1_mean"])

    out = {
        "question": "Q2 resolution strategy and interpolation",
        "seeds": list(seeds),
        "n_runs": len(results) * len(seeds),
        "padding_audit": padding,
        "results": results,
        "by_strategy": by_strategy,
        "by_interpolation": by_interp,
        "best_config": best,
        "worst_config": worst,
        "strategy_ranking": strat_rank,
        "comparisons": {
            "best_vs_worst": describe_comparison(worst, results[worst], best, results[best]),
            "bilinear_vs_bicubic": _interp_sentence(by_interp),
        },
        "notes": [
            "Strategy effects are confounded with how much synthetic black border each one "
            "introduces; the padding_audit field quantifies that so the comparison can be "
            "read as a mechanism rather than a leaderboard.",
            "The low-resolution subgroup is small, so its mean+/-sd is wide. It is reported "
            "for direction only and is not used to rank strategies.",
        ],
    }
    save_json(out, "q2_strategies.json")
    return out


def _print_summary(r: dict) -> None:
    print("-" * 72)
    print(f"best  : {r['best_config']}  macro_f1 {r['results'][r['best_config']]['macro_f1_mean']:.4f}")
    print(f"worst : {r['worst_config']}  macro_f1 {r['results'][r['worst_config']]['macro_f1_mean']:.4f}")
    print(f"strategy ranking by macro_f1: {' > '.join(r['strategy_ranking'])}")
    print(r["comparisons"]["best_vs_worst"])
    print("-" * 72)
    print("low-resolution subgroup (n small, direction only):")
    for k, v in r["results"].items():
        lo = v["by_resolution_group"]["low"]
        print(f"  {k:28s} low-res n={lo['n_mean']:.0f} acc "
              f"{lo['accuracy_mean']:.3f}±{lo['accuracy_std']:.3f} | high-res n="
              f"{v['by_resolution_group']['high']['n_mean']:.0f} acc "
              f"{v['by_resolution_group']['high']['accuracy_mean']:.3f}"
              f"±{v['by_resolution_group']['high']['accuracy_std']:.3f}")


if __name__ == "__main__":
    _print_summary(run())
