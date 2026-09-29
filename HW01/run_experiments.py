"""HW01 entry point: runs Q1, Q2, Q3 and renders every figure.

    python HW01/run_experiments.py              # everything
    python HW01/run_experiments.py --stage q1   # one stage
    python HW01/run_experiments.py --no-plots   # skip figures

Model lineage for the whole term:
    M0  real images only, no custom augmentation
    M1  M0 + the custom augmentation chosen in Q3      <- this submission
    M2  M1 + the improvement proposed in HW02
    M3  M2 + diffusion-generated data                 <- HW03
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data import N_VAL_PER_CLASS, PLOTS_DIR, RESULTS_DIR, SEEDS, ensure_dirs, set_global_seed
from metrics import format_pm

CLASS_LABELS = [f"cl{i:02d}" for i in range(1, 21)]


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_q1(r: dict) -> None:
    plt = _plt()
    prof = r["dataset_profile"]
    ar = prof["aspect_ratio"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))

    ax = axes[0]
    sr = r["split_report"]["seeds"]
    seeds = list(sr)
    ntr = [sr[s]["n_train"] for s in seeds]
    nva = [sr[s]["n_val"] for s in seeds]
    ax.bar(np.arange(len(seeds)) - 0.2, ntr, 0.4, label="train")
    ax.bar(np.arange(len(seeds)) + 0.2, nva, 0.4, label="val")
    for i, (a, b) in enumerate(zip(ntr, nva)):
        ax.text(i - 0.2, a + 2, str(a), ha="center", fontsize=8)
        ax.text(i + 0.2, b + 2, str(b), ha="center", fontsize=8)
    ax.set_xticks(np.arange(len(seeds)), [f"seed {s}" for s in seeds])
    ax.set_ylabel("images")
    ax.set_title(f"Q1 split sizes ({N_VAL_PER_CLASS} images per class held out)")
    ax.legend()
    ax.set_ylim(0, max(ntr) * 1.15)

    ax = axes[1]
    m = np.asarray(r["normalisation"]["per_strategy"]["resize_crop"]["mean"])
    s = np.asarray(r["normalisation"]["per_strategy"]["resize_crop"]["std"])
    ch = ["R", "G", "B"]
    x = np.arange(3)
    ax.bar(x - 0.2, m, 0.4, label="dataset mean")
    ax.bar(x + 0.2, [0.485, 0.456, 0.406], 0.4, label="ImageNet mean", alpha=0.6)
    ax.set_xticks(x, ch); ax.set_title("Q1 mean: dataset vs ImageNet"); ax.legend()
    ax2 = ax.twinx()
    ax2.plot(x, s, "o-", color="crimson", label="dataset std")
    ax2.plot(x, [0.229, 0.224, 0.225], "s--", color="gray", label="ImageNet std")
    ax2.set_ylim(0, 0.6); ax2.set_ylabel("std")

    ax = axes[2]
    ax.hist(np.log10([prof["short_side"]["min"], prof["short_side"]["median"],
                      prof["short_side"]["max"]]), bins=3, alpha=0)
    ax.bar(["min", "median", "max"], [prof["short_side"]["min"], prof["short_side"]["median"],
                                      prof["short_side"]["max"]], color="steelblue", alpha=0.8)
    ax.axhline(224, color="crimson", ls="--", label="224px input size")
    ax.set_ylabel("short side (px)"); ax.legend()
    ax.set_title("Q2 relevance: short side vs input size")

    fig.suptitle("Q1: reproducible input pipeline", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "q1_pipeline.png"), dpi=150)
    plt.close(fig)


def plot_q2(r: dict) -> None:
    plt = _plt()
    labels = list(r["results"])
    f1 = [r["results"][k]["macro_f1_mean"] for k in labels]
    sd = [r["results"][k]["macro_f1_std"] for k in labels]

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))

    ax = axes[0]
    y = np.arange(len(labels))
    ax.barh(y, f1, xerr=sd, color=["#4c72b0" if "direct" in k else
                                    "#55a868" if "resize_crop" in k else "#c44e52"
                                    for k in labels], alpha=0.85, capsize=3)
    ax.set_yticks(y, [k.replace("|", "\n") for k in labels], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Macro-F1 (mean ± sd over 3 seeds)")
    ax.set_title("Q2: strategy × interpolation")
    best = r["best_config"]
    ax.axvline(r["results"][best]["macro_f1_mean"], color="crimson", ls=":", alpha=0.7)

    ax = axes[1]
    pa = r["padding_audit"]
    strat = list(pa)
    pct = [pa[s]["pct_with_synthetic_black"] for s in strat]
    up = [pa[s]["median_short_side_upscale"] for s in strat]
    x = np.arange(len(strat))
    ax.bar(x - 0.2, pct, 0.4, label="% images w/ synthetic black")
    ax.set_ylabel("% of images with black border", color="tab:blue")
    ax.set_xticks(x, strat, rotation=15)
    ax2 = ax.twinx()
    ax2.bar(x + 0.2, up, 0.4, color="tab:orange", alpha=0.7, label="median short-side upscale")
    ax2.set_ylabel("median upscale factor (×)", color="tab:orange")
    ax.set_title("Q2 mechanism: what each strategy fabricates")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8)

    ax = axes[2]
    width = 0.35
    for j, g in enumerate(("high", "low")):
        m = [r["results"][k]["by_resolution_group"][g]["accuracy_mean"] for k in labels]
        s = [r["results"][k]["by_resolution_group"][g]["accuracy_std"] for k in labels]
        ax.barh(np.arange(len(labels)) + (j - 0.5) * width, m, height=width,
                xerr=s, alpha=0.85, capsize=2, label=f"{g}-res")
    ax.set_yticks(np.arange(len(labels)), labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Top-1 accuracy on validation subgroup")
    ax.set_title("Q2: high- vs low-resolution subgroup\n(low group is small — direction only)")
    ax.legend(fontsize=8)

    fig.suptitle("Q2: resolution strategy comparison", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "q2_strategies.png"), dpi=150)
    plt.close(fig)


def plot_confusion(agg: dict, class_names: list[str], title: str, filename: str) -> None:
    plt = _plt()
    cm = np.asarray(agg["mean_normalised_confusion_matrix"])
    fig, ax = plt.subplots(figsize=(9.5, 8.2))
    im = ax.imshow(cm, cmap="magma_r", vmin=0, vmax=1)
    ax.set_xticks(range(len(class_names)), class_names, rotation=90, fontsize=7)
    ax.set_yticks(range(len(class_names)), class_names, fontsize=7)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if cm[i, j] > 0.34:
                ax.text(j, i, f"{cm[i,j]:.2f}", ha="center", va="center",
                        fontsize=5.5, color="white")
    fig.colorbar(im, ax=ax, shrink=0.8, label="row-normalised rate (mean of 3 seeds)")
    ax.set_title(title, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, filename), dpi=150)
    plt.close(fig)


def plot_per_class(m0: dict, m1: dict) -> None:
    plt = _plt()
    a = np.asarray(m0["per_class_recall_mean"])
    a_sd = np.asarray(m0["per_class_recall_std"])
    b = np.asarray(m1["per_class_recall_mean"])
    x = np.arange(len(CLASS_LABELS))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), sharex=True)

    ax1.bar(x - 0.2, a, 0.4, yerr=a_sd, capsize=2, label="M0", color="#4c72b0", alpha=0.9)
    ax1.bar(x + 0.2, b, 0.4, label="M1 (best recipe)", color="#c44e52", alpha=0.9)
    ax1.axhline(m0["worst_class_recall_mean"], color="gray", ls=":", label="M0 worst-class mean")
    ax1.set_ylabel("per-class recall (mean of 3 seeds)")
    ax1.set_title("Per-class recall, M0 vs M1 (error bars: sd over 3 seeds)", fontweight="bold")
    ax1.legend()

    d = b - a
    colors = ["#c44e52" if v < -1e-9 else "#55a868" for v in d]
    ax2.bar(x, d, color=colors)
    ax2.axhline(0, color="k", lw=0.8)
    ax2.set_xticks(x, CLASS_LABELS, rotation=90, fontsize=8)
    ax2.set_ylabel("Δ recall (M1 − M0)")
    ax2.set_title("Where the targeted augmentation moved recall", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "per_class_recall_m0_vs_m1.png"), dpi=150)
    plt.close(fig)


def plot_ratio_sweep(r: dict) -> None:
    plt = _plt()
    pts = r["turning_points"]
    recipes = list(pts)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    ax = axes[0]
    for name in recipes:
        p = pts[name]
        ratios = p["ratios"]
        f1 = np.asarray(p["macro_f1"]); sd = np.asarray(p["macro_f1_std"])
        wr = np.asarray(p["worst_class_recall"])
        ax.errorbar(ratios, f1, yerr=sd, marker="o", capsize=3, label=f"{name} Macro-F1")
        ax.plot(ratios, wr, marker="s", ls="--", alpha=0.55, label=f"{name} worst-recall")
        if p["turning_point"]:
            tp = p["turning_point"]["ratio"]
            ax.axvline(tp, color="crimson", ls=":", alpha=0.35)
    ax.axhline(r["m0"]["macro_f1_mean"], color="k", ls="--", lw=1,
               label=f"M0 {r['m0']['macro_f1_mean']:.3f}")
    ax.set_xscale("log", base=2)
    tick_ratios = sorted({r for p in pts.values() for r in (p["ratios"][0], p["best_ratio"])})
    ax.set_xticks(tick_ratios)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel("per-class augmentation ratio on targeted classes (×)")
    ax.set_ylabel("score")
    ax.set_title("Q3: ratio sweep — Macro-F1 and worst-class recall", fontweight="bold")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)

    ax = axes[1]
    for name, r in r["semantic_integrity"].items():
        c = r["cosine_to_source_mean"]
        a = r["prototype_agreement"]
        if c is None:
            continue
        ax.scatter([c], [a if a is not None else np.nan], s=140, label=name)
        ax.annotate(name, (c, a if a is not None else 0), fontsize=8,
                    textcoords="offset points", xytext=(6, 4))
    ax.axhline(0.9, color="gray", ls=":", label="agreement 0.9")
    ax.axvline(0.85, color="gray", ls=":", label="cosine 0.85")
    ax.set_xlabel("mean cosine to source image (feature space)")
    ax.set_ylabel("prototype agreement (nearest centroid = true class)")
    ax.set_xlim(0.4, 1.02); ax.set_ylim(0, 1.05)
    ax.set_title("Q3: semantic integrity of each recipe", fontweight="bold")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "q3_augmentation_sweep.png"), dpi=150)
    plt.close(fig)


def plot_augmentation_samples(q3_result: dict | None = None) -> None:
    """Visual check that the recipes did not destroy the images.

    Semantics are partly a visual property, so the assignment's "check whether
    augmentation broke the image" is answered with a montage a human can read
    alongside the feature-space numbers.

    When the Q3 result is available the rows are the *hardest* class (lowest M0
    recall), a median class, and the *easiest* class, so the reader can see
    whether a recipe damages easy classes more than hard ones.
    """
    plt = _plt()
    import random

    from augmentation import SPECS, apply_augmentation
    from data import make_split, pil_loader

    split = make_split(SEEDS[0])
    if q3_result is not None:
        pcr = np.asarray(q3_result["m0"]["per_class_recall_mean"])
        ranked = [int(i) for i in np.argsort(pcr)]  # ascending recall
        weak = ranked[0]
        mid = ranked[len(ranked) // 2]
        strong = ranked[-1]
    else:
        sizes = sorted(range(20), key=lambda c: -sum(1 for _, y in split.train if y == c))
        weak, mid, strong = sizes[0], sizes[len(sizes) // 2], sizes[-1]
    picks = [weak, mid, strong]

    recipes = [n for n in SPECS if n != "none"]
    n_show = 3

    fig, axes = plt.subplots(len(picks) * (len(recipes) + 1), n_show,
                             figsize=(3 * n_show, 2.1 * len(picks) * (len(recipes) + 1)),
                             squeeze=False)
    axes = axes.ravel()  # flat view: one entry per cell
    rng = random.Random(0)
    row = 0
    for c in picks:
        paths = [p for p, y in split.train if y == c][:n_show]
        for k, p in enumerate(paths):
            axes[row + k].imshow(np.asarray(pil_loader(p)))
        axes[row].set_ylabel(f"cl{weak+1:02d} (weak)", fontsize=9, fontweight="bold")
        row += n_show
        for rname in recipes:
            for k, p in enumerate(paths):
                axes[row + k].imshow(np.asarray(apply_augmentation(pil_loader(p), SPECS[rname], rng)))
            axes[row].set_ylabel(rname, fontsize=8)
            row += n_show
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Q3: augmentation recipes on weak / mid / strong classes "
                 "(column = image, block = recipe)", fontweight="bold", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(os.path.join(PLOTS_DIR, "q3_augmentation_samples.png"), dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description="HW01: few-shot input pipeline and augmentation")
    ap.add_argument("--stage", choices=["q1", "q2", "q3", "all"], default="all")
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--no-samples", action="store_true",
                    help="skip the augmentation montage (slowest figure)")
    args = ap.parse_args()

    ensure_dirs()
    set_global_seed(SEEDS[0])
    t0 = time.time()
    print(f"HW01  seeds={SEEDS}  stages={args.stage}")
    print(f"python {sys.version.split()[0]}  results -> {RESULTS_DIR}\n")

    store = {}
    if args.stage in ("q1", "all"):
        import q1_input_pipeline as q1
        store["q1"] = q1.run()
        q1._print_summary(store["q1"])
        print()
        if not args.no_plots:
            plot_q1(store["q1"])

    if args.stage in ("q2", "all"):
        import q2_resolution_comparison as q2
        store["q2"] = q2.run()
        q2._print_summary(store["q2"])
        print()
        if not args.no_plots:
            plot_q2(store["q2"])

    if args.stage in ("q3", "all"):
        import q3_augmentation_design as q3
        store["q3"] = q3.run()
        q3._print_summary(store["q3"])
        if not args.no_plots:
            plot_confusion(store["q3"]["m0"], CLASS_LABELS,
                           "M0 baseline — confusion matrix (row-normalised, mean of 3 seeds)",
                           "m0_confusion_matrix.png")
            plot_confusion(store["q3"]["best_m1"], CLASS_LABELS,
                           f"M1 ({store['q3']['best_recipe']}) — confusion matrix "
                           "(row-normalised, mean of 3 seeds)",
                           "m1_confusion_matrix.png")
            plot_per_class(store["q3"]["m0"], store["q3"]["best_m1"])
            plot_ratio_sweep(store["q3"])
            if not args.no_samples:
                plot_augmentation_samples()

    print(f"\n{'=' * 72}")
    print("Headline numbers (mean ± sd over 3 sampling configurations)")
    print("=" * 72)
    if "q3" in store:
        m0, m1 = store["q3"]["m0"], store["q3"]["best_m1"]
        print(f"  M0  Top-1 {format_pm(m0['top1_accuracy_mean'], m0['top1_accuracy_std'])}  "
              f"Macro-F1 {format_pm(m0['macro_f1_mean'], m0['macro_f1_std'])}  "
              f"worst {format_pm(m0['worst_class_recall_mean'], m0['worst_class_recall_std'])}")
        print(f"  M1  Top-1 {format_pm(m1['top1_accuracy_mean'], m1['top1_accuracy_std'])}  "
              f"Macro-F1 {format_pm(m1['macro_f1_mean'], m1['macro_f1_std'])}  "
              f"worst {format_pm(m1['worst_class_recall_mean'], m1['worst_class_recall_std'])}")
    if "q2" in store:
        print(f"  Q2 best geometry: {store['q2']['best_config']} "
              f"Macro-F1 {format_pm(store['q2']['results'][store['q2']['best_config']]['macro_f1_mean'], store['q2']['results'][store['q2']['best_config']]['macro_f1_std'])}")
    print(f"\nartefacts: {RESULTS_DIR}/*.json, {PLOTS_DIR}/*.png")
    print(f"total wall time {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
