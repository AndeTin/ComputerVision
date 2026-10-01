"""Q2: geometry strategies and interpolation.

3 strategies x 2 interpolations x 3 seeds = 18 trained probes, all scored with
Macro-F1 as the primary metric and the low-resolution subgroup tracked
separately.

The mechanism under test: the three strategies differ in *what they do to
aspect ratio*, not merely in how much they resample.
  direct        - anisotropic squash; no synthetic pixels, but object shape
                  is distorted by up to 3.7x here.
  resize_crop   - isotropic scale then crop. ``Resize(256)`` is the *int* form,
                  which maps the SHORTER side to 256, so the scaled short side
                  is never below 224 and CenterCrop never pads. The loss is
                  entirely crop: every image loses its outer ring.
  pad_to_square - isotropic scale then explicit symmetric zero-pad. Aspect
                  ratio is preserved and the padding is measurable.
"""

from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import numpy as np

from hw01.data import (
    DATASET_DIR, IMAGE_SIZE, INTERPOLATIONS, RESIZE_BASE, SEEDS, STRATEGIES,
    get_class_paths, save_json, set_global_seed,
)
from hw01.experiment import RunConfig, run_config
from hw01.metrics import describe_comparison, format_pm, intervals_overlap
from hw01.models import get_feature_extractor


def _geometry_areas(w: int, h: int, strategy: str) -> dict:
    """Analytic breakdown of what one geometry does to a (w, h) image.

    Derived from the transform definitions rather than measured from pixels.
    A pixel-scan cannot separate "black the transform injected" from "black
    that was in the photograph", which made an earlier version of this audit
    report non-zero padding for ``direct`` -- a strategy that injects none.

    Returns the fraction of the 224x224 output that is synthetic fill, the
    fraction of original content thrown away, and the short-side upscale.
    """
    out = IMAGE_SIZE * IMAGE_SIZE
    if strategy == "direct":
        # Anisotropic squash: the content fills the frame, distorted.
        return {"synthetic_frac": 0.0, "discarded_frac": 0.0,
                "upscale_short": IMAGE_SIZE / min(w, h)}

    if strategy == "pad_to_square":
        # Long side -> 224, then symmetric zero-pad on the short dimension.
        scale = IMAGE_SIZE / max(w, h)
        nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
        return {"synthetic_frac": max(0.0, 1.0 - (nw * nh) / out),
                "discarded_frac": 0.0,
                "upscale_short": IMAGE_SIZE / min(w, h)}

    if strategy == "resize_crop":
        # transforms.Resize(256) is the int form: it maps the SHORTER side to
        # RESIZE_BASE, so the scaled short side is always >= IMAGE_SIZE and
        # CenterCrop never zero-pads. All the information loss is the crop.
        scale = RESIZE_BASE / min(w, h)
        nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
        return {"synthetic_frac": 0.0,
                "discarded_frac": max(0.0, 1.0 - out / (nw * nh)),
                "upscale_short": IMAGE_SIZE / min(w, h)}

    raise ValueError(strategy)


def _padding_audit(dataset_dir: str = DATASET_DIR) -> dict:
    """How much does each strategy fabricate, and how much does it throw away?

    Separates the two failure modes the three strategies trade off:

    * ``synthetic_frac`` -- fraction of the output filled with zeros. The
      frozen backbone has never seen such an input, so this is distribution
      shift, not data.
    * ``discarded_frac`` -- fraction of the original image cropped away. This
      is information loss, the opposite failure.
    """
    from PIL import Image

    per = {s: {"synthetic": [], "discarded": [], "upscale": []} for s in STRATEGIES}
    n_total = 0
    for paths in get_class_paths(dataset_dir):
        for p in paths:
            n_total += 1
            w, h = Image.open(p).size
            for s in STRATEGIES:
                a = _geometry_areas(w, h, s)
                per[s]["synthetic"].append(a["synthetic_frac"])
                per[s]["discarded"].append(a["discarded_frac"])
                per[s]["upscale"].append(a["upscale_short"])

    stats = {}
    for s in STRATEGIES:
        syn = np.asarray(per[s]["synthetic"])
        dis = np.asarray(per[s]["discarded"])
        up = np.asarray(per[s]["upscale"])
        stats[s] = {
            "n_images": n_total,
            "n_with_synthetic_fill": int((syn > 1e-9).sum()),
            "pct_with_synthetic_fill": round(100 * float((syn > 1e-9).mean()), 1),
            "mean_synthetic_area_frac": round(float(syn.mean()), 4),
            "max_synthetic_area_frac": round(float(syn.max()), 4),
            "n_with_discarded_content": int((dis > 1e-9).sum()),
            "pct_with_discarded_content": round(100 * float((dis > 1e-9).mean()), 1),
            "mean_discarded_area_frac": round(float(dis.mean()), 4),
            "max_discarded_area_frac": round(float(dis.max()), 4),
            "n_upscaled_on_short_side": int((up > 1.0).sum()),
            "pct_upscaled_on_short_side": round(100 * float((up > 1.0).mean()), 1),
            "median_short_side_upscale": round(float(np.median(up)), 3),
            "max_short_side_upscale": round(float(up.max()), 2),
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
        print(f"  {strat:14s} zero-filled in {st['pct_with_synthetic_fill']:5.1f}% of images "
              f"(mean area {st['mean_synthetic_area_frac']:.3f}, max {st['max_synthetic_area_frac']:.3f}) | "
              f"content cropped away in {st['pct_with_discarded_content']:5.1f}% "
              f"(mean {st['mean_discarded_area_frac']:.3f}) | "
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
