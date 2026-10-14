"""HW02 entry point: the four M1 diagnostics, the M2 study, and every figure.

    python HW02/run_hw02.py                 # diagnostics + M2 study + plots
    python HW02/run_hw02.py --stage diag1   # one diagnostic
    python HW02/run_hw02.py --stage m2      # M2 variants + comparison
    python HW02/run_hw02.py --no-plots      # numbers only

Writes JSON under ``HW02/results/`` and figures under ``HW02/plots/``; the frozen
feature cache is the shared ``HW01/.cache`` produced by stage 1.  Every reported
number is a mean +/- standard deviation over the three sampling configurations
(seeds 42, 420, 4200), and differences are phrased only by interval overlap.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

HW02_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HW02_DIR)
HW01_DIR = os.path.join(REPO_ROOT, "HW01")
for _p in (HW01_DIR, HW02_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from hw01.data import SEEDS, pil_loader  # noqa: E402
from hw01.metrics import format_pm  # noqa: E402
from hw01.models import get_feature_extractor  # noqa: E402

from hw02_diagnostics import (  # noqa: E402
    diag1_error_samples, diag2_learning_curve, diag3_train_val_gap,
    diag4_feature_space,
)
from hw02_m2_models import (  # noqa: E402
    CLASS_NAMES, HW02_FULL_RESULTS_DIR, HW02_PLOTS_DIR, HW02_RESULTS_DIR,
    PROBLEM_CLASSES, ensure_hw02_dirs, get_device, run_m2_study, save_json,
)

# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_diag1(d: dict, max_pairs: int = 6) -> str:
    plt = _plt()
    from matplotlib.gridspec import GridSpec

    pairs = [p for p in d["pairs"] if p.get("predicted_class_prototype_train_image")][:max_pairs]
    n = max(len(pairs), 1)
    fig = plt.figure(figsize=(13.5, max(3.6, 2.1 * n)))
    gs = GridSpec(n, 3, figure=fig, width_ratios=[1.2, 1.0, 1.0])

    axcm = fig.add_subplot(gs[:, 0])
    cm = np.asarray(d["mean_normalised_confusion_matrix"])
    names = d["class_names"]
    im = axcm.imshow(cm, cmap="magma_r", vmin=0, vmax=1)
    axcm.set_xticks(range(len(names)), names, rotation=90, fontsize=6)
    axcm.set_yticks(range(len(names)), names, fontsize=6)
    axcm.set_xlabel("predicted"); axcm.set_ylabel("true")
    axcm.set_title("M1 confusion matrix\n(row-normalised, mean of 3 seeds)",
                   fontsize=9, fontweight="bold")
    fig.colorbar(im, ax=axcm, shrink=0.45, label="rate")

    for r, p in enumerate(pairs):
        ex = p["example"]
        pr = p["predicted_class_prototype_train_image"]
        a1 = fig.add_subplot(gs[r, 1])
        a1.imshow(np.asarray(pil_loader(ex["path"])))
        a1.set_title(f"{p['true_name']} -> {p['pred_name']} "
                     f"(conf {ex['confidence']:.2f}, {p['n_seeds']}/3 seeds)", fontsize=9)
        a2 = fig.add_subplot(gs[r, 2])
        a2.imshow(np.asarray(pil_loader(pr["path"])))
        a2.set_title(f"typical {p['pred_name']} train image (cos {pr['cosine']:.2f})", fontsize=9)
        for a in (a1, a2):
            a.set_xticks([]); a.set_yticks([])

    diff = sum(s["n_errors"] for s in d["per_seed"])
    fig.suptitle(f"HW02 diag1: M1 error samples ({diff} validation errors across 3 seeds)",
                 fontweight="bold", y=1.0)
    fig.tight_layout()
    path = os.path.join(HW02_PLOTS_DIR, "diag1_error_samples.png")
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_diag2(d: dict) -> str:
    plt = _plt()
    xs = np.arange(len(d["k_values"]))
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for m, color in (("M0", "#4c72b0"), ("M1", "#c44e52")):
        c = d["curve"][m]
        ax.errorbar(xs, c["macro_f1_mean"], yerr=c["macro_f1_std"], marker="o",
                    capsize=4, color=color, label=f"{m} Macro-F1")
        ax2.errorbar(xs, c["top1_accuracy_mean"], yerr=c["top1_accuracy_std"], marker="s",
                     capsize=4, color=color, label=f"{m} Top-1")
    for a, title, ylab in ((ax, "Macro-F1", "Macro-F1 (mean \u00b1 sd)"),
                           (ax2, "Top-1 accuracy", "Top-1 accuracy (mean \u00b1 sd)")):
        a.set_xticks(xs, [str(k) for k in d["k_values"]])
        a.set_xlabel("per-class training cap (originals)")
        a.set_ylabel(ylab)
        a.set_title(f"HW02 diag2: learning curve \u2014 {title}", fontweight="bold")
        a.grid(alpha=0.3); a.legend(fontsize=8)
    ntr = d["curve"]["M0"]["n_train_effective"]
    for x, n in zip(xs, ntr):
        ax.annotate(f"~{n:.0f}", (x, ax.get_ylim()[0]), fontsize=7, ha="center",
                    va="bottom", color="gray")
    fig.tight_layout()
    path = os.path.join(HW02_PLOTS_DIR, "diag2_learning_curve.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_diag3(d: dict) -> str:
    plt = _plt()
    metrics = [("top1_accuracy", "Top-1 accuracy"), ("macro_f1", "Macro-F1"),
               ("log_loss", "cross-entropy loss")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, (key, title) in zip(axes, metrics):
        labels, tr, trs, va, vas = [], [], [], [], []
        for m in ("M0", "M1"):
            labels.append(m)
            tr.append(d[m][f"train_{key}"]["mean"]); trs.append(d[m][f"train_{key}"]["std"])
            va.append(d[m][f"val_{key}"]["mean"]); vas.append(d[m][f"val_{key}"]["std"])
        x = np.arange(len(labels))
        ax.bar(x - 0.2, tr, 0.4, yerr=trs, capsize=3, label="train", color="#8c8c8c")
        ax.bar(x + 0.2, va, 0.4, yerr=vas, capsize=3, label="val", color="#4c72b0")
        ax.set_xticks(x, labels)
        ax.set_title(f"{title}\n(gap: M0 {d['M0']['gaps'][_gapkey(key)]:+.4f}, "
                     f"M1 {d['M1']['gaps'][_gapkey(key)]:+.4f})", fontweight="bold", fontsize=10)
        ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    fig.suptitle("HW02 diag3: train vs validation gap (over- or under-fitting?)",
                 fontweight="bold")
    fig.tight_layout()
    path = os.path.join(HW02_PLOTS_DIR, "diag3_train_val_gap.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def _gapkey(key: str) -> str:
    return {"top1_accuracy": "top1", "macro_f1": "macro_f1", "log_loss": "log_loss"}[key]


def plot_diag4(d: dict) -> str:
    plt = _plt()
    pts = d["points"]
    use_umap = pts.get("umap") is not None
    methods = [("tsne", "t-SNE")] + ([("umap", "UMAP")] if use_umap else [])
    labels = np.asarray(pts["label"])
    cat = np.asarray(pts["category"])
    is_prob = np.asarray(pts["is_problem_class"])
    cmap = plt.get_cmap("tab20")

    fig, axes = plt.subplots(len(methods), 2, figsize=(14, 6.0 * len(methods)), squeeze=False)
    for row, (key, nice) in enumerate(methods):
        xy = np.asarray(pts[key])

        ax = axes[row, 0]
        for c in range(len(CLASS_NAMES)):
            m = labels == c
            if not m.any():
                continue
            ax.scatter(xy[m, 0], xy[m, 1], s=14, color=cmap(c % 20),
                       marker="o", alpha=0.75, linewidths=0)
        ax.set_title(f"HW02 diag4 ({nice}): all 20 classes", fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])

        ax = axes[row, 1]
        styles = {
            "original_train": dict(marker="o", s=30, facecolors="none", edgecolors="k", label="original train"),
            "augmented_train": dict(marker="x", s=34, c="tab:green", label="M1 augmented"),
            "val_correct": dict(marker="s", s=30, c="tab:blue", label="val correct"),
            "val_wrong": dict(marker="*", s=170, c="crimson", label="val misclassified"),
        }
        for name, st in styles.items():
            m = is_prob & (cat == name)
            if m.any():
                ax.scatter(xy[m, 0], xy[m, 1], **st)
        for c in PROBLEM_CLASSES:
            m = is_prob & (labels == c)
            if m.any():
                ax.annotate(CLASS_NAMES[c], (xy[m, 0].mean(), xy[m, 1].mean()),
                            fontsize=9, fontweight="bold")
        ax.set_title(f"HW02 diag4 ({nice}): problem classes zoom", fontweight="bold")
        ax.legend(fontsize=8, loc="best")
        ax.set_xticks([]); ax.set_yticks([])

    fig.tight_layout()
    path = os.path.join(HW02_PLOTS_DIR, "diag4_tsne_umap.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_comparison(models: dict, chosen: str) -> str:
    plt = _plt()
    names = list(models)
    f1 = [models[k]["macro_f1_mean"] for k in names]
    f1s = [models[k]["macro_f1_std"] for k in names]
    wr = [models[k]["worst_class_recall_mean"] for k in names]
    gaps = [models[k]["loss_gap_mean"] for k in names]
    x = np.arange(len(names))
    colors = ["crimson" if k == chosen else "#4c72b0" for k in names]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(16, 5.4), gridspec_kw={"width_ratios": [2, 1]})
    ax.bar(x - 0.2, f1, 0.4, yerr=f1s, capsize=3, color=colors, label="Macro-F1")
    ax.bar(x + 0.2, wr, 0.4, color="#8c8c8c", label="worst-class recall")
    ax.set_xticks(x, [n.replace("(", "\n(") for n in names], fontsize=8)
    ax.set_ylabel("score (mean \u00b1 sd)")
    ax.set_title("HW02: M0 / M1 / M2 candidates", fontweight="bold")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")

    ax2.bar(x, gaps, color=colors)
    ax2.set_xticks(x, names, rotation=90, fontsize=7)
    ax2.set_ylabel("val - train cross-entropy")
    ax2.set_title("train-val loss gap", fontweight="bold")
    ax2.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    path = os.path.join(HW02_PLOTS_DIR, "m0_m1_m2_comparison.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description="HW02: second-stage diagnostics and M2")
    ap.add_argument("--stage", choices=["diag1", "diag2", "diag3", "diag4", "m2", "all"],
                    default="all")
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--device", default="cpu",
                    help="feature-extraction device; 'cpu' reproduces the stage-1 "
                         "cache bit-for-bit, 'cuda' is faster but shifts features "
                         "at the ~1e-3 level and can flip a borderline sample")
    ap.add_argument("--seed-diagnostic", type=int, default=SEEDS[0],
                    help="seed used for the qualitative feature-space map")
    args = ap.parse_args()

    ensure_hw02_dirs()
    device = args.device or get_device()
    extractor = get_feature_extractor(device)
    t0 = time.time()
    produced: list[str] = []

    print(f"HW02  seeds={SEEDS}  device={device}  stage={args.stage}")
    print(f"python {sys.version.split()[0]}  results -> {HW02_RESULTS_DIR}\n")

    def emit(json_name, payload, plot_fn=None):
        produced.append(save_json(payload, json_name))
        if plot_fn and not args.no_plots:
            p = plot_fn(payload)
            if p:
                produced.append(p)

    if args.stage in ("diag1", "all"):
        d1 = diag1_error_samples(seeds=SEEDS, extractor=extractor, device=device)
        emit("diag1_error_samples.json", d1, plot_diag1)
        top = d1["pairs"][0] if d1["pairs"] else None
        if top:
            print(f"  diag1: most stable error {top['true_name']}->{top['pred_name']} "
                  f"({top['n_seeds']}/3 seeds, {top['n_occurrences']} occurrences)\n")

    if args.stage in ("diag2", "all"):
        d2 = diag2_learning_curve(seeds=SEEDS, extractor=extractor, device=device)
        emit("diag2_learning_curve.json", d2, plot_diag2)
        m1 = d2["curve"]["M1"]["macro_f1_mean"]
        print(f"  diag2: M1 Macro-F1 vs cap -> {[round(v, 4) for v in m1]}\n")

    if args.stage in ("diag3", "all"):
        d3 = diag3_train_val_gap(seeds=SEEDS, extractor=extractor, device=device)
        emit("diag3_train_val_gap.json", d3, plot_diag3)
        print(f"  diag3: M1 train-val acc gap {d3['M1']['gaps']['top1']:+.4f}, "
              f"loss gap {d3['M1']['gaps']['log_loss']:+.4f}\n")

    if args.stage in ("diag4", "all"):
        d4 = diag4_feature_space(seed=args.seed_diagnostic, extractor=extractor, device=device)
        emit("diag4_feature_space.json", d4, plot_diag4)
        print(f"  diag4: embedding via {d4['embedding_methods']}, "
              f"{len(d4['problem_classes'])} problem classes\n")

    if args.stage in ("m2", "all"):
        study = run_m2_study(seeds=SEEDS, extractor=extractor, device=device)
        emit("m2_variants.json", {
            "question": "Which single change to M1 most improves the frozen-feature head?",
            "geometry": study["geometry"], "m1_recipe": study["m1_recipe"],
            "variant_a": study["variant_a"], "variant_b": study["variant_b"],
            "variant_c": study["variant_c"],
            "best": {"variant_a": study["m2"]["variant_a_best"],
                     "variant_b": study["m2"]["variant_b_best"],
                     "variant_c": study["m2"]["variant_c_best"]},
            "note": "selection is on the same validation set that is reported (3 seeds)",
        })
        comparison = {
            "question": "M0 vs M1 vs M2 on direct + bilinear features",
            "seeds": list(SEEDS),
            "geometry": study["geometry"], "m1_recipe": study["m1_recipe"],
            "models": study["models"],
            "comparison": study["comparison"],
            "chosen_m2": study["m2"]["key"],
            "reporting_rule": ("comparisons use +/-1 sd interval overlap; three seeds "
                               "cannot support a significance claim"),
        }
        emit("m0_m1_m2_comparison.json", comparison,
             lambda _p: plot_comparison(study["models"], study["m2"]["key"]))
        produced.append(save_json({
            "generated_seeds": list(SEEDS),
            "m2": study["m2"]["key"],
            "headline": {k: {"macro_f1": format_pm(v["macro_f1_mean"], v["macro_f1_std"]),
                             "top1": format_pm(v["top1_accuracy_mean"], v["top1_accuracy_std"]),
                             "worst_class_recall_mean": v["worst_class_recall_mean"]}
                         for k, v in study["models"].items()},
            "comparison_table": study["comparison"]["table"],
        }, "hw02_summary.json", full=False))
        print(f"  m2: chosen {study['m2']['key']}  "
              f"Macro-F1 {format_pm(study['m2']['block']['macro_f1_mean'], study['m2']['block']['macro_f1_std'])}\n")

    print("=" * 72)
    print("HW02 artefacts")
    print("=" * 72)
    for p in produced:
        print(f"  {p}")
    print(f"total wall time {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
