"""Q3: custom few-shot augmentation, producing M0 and M1.

Sequence:
  1. M0 baseline -- frozen ResNet-18 + LogisticRegression, no custom
     augmentation, on the geometry chosen in Q2. Confusion matrix and
     per-class recall identify the weak tail.
  2. Semantic integrity of the augmentation recipes, measured on features
     before any model is trained, so a destructive recipe is caught early.
  3. M1 -- augmentation applied *only* to the weak classes, sweeping the
     per-class augmentation ratio to locate the turning point.
  4. Failure-case inspection: which weak classes never recover, and what the
     augmented samples actually look like.
"""

from __future__ import annotations

import numpy as np

from augmentation import SPECS, AugmentSpec
from data import SEEDS, ensure_dirs, save_json, set_global_seed
from experiment import RunConfig, run_config
from metrics import describe_comparison, format_pm, intervals_overlap, rank_weak_classes
from models import get_feature_extractor

#: Per-class augmentation ratios swept to find the turning point.
#: 1.0 is M0 (no augmentation); the rest add copies to the weak classes only.
RATIO_SWEEP = (1.0, 2.0, 4.0, 8.0, 16.0)

#: How many classes count as "weak" and receive targeted augmentation.
N_TARGET_CLASSES = 5

#: Geometry carried over from Q2 so Q3 isolates the augmentation effect.
#: Read from results/q2_strategies.json when that stage has been run, so the
#: winner is never hardcoded and Q2/Q3 cannot silently disagree. The fallback
#: matches Q2's default sweep order and is only used if Q2 has not been run.
_FALLBACK_GEOMETRY = ("resize_crop", "bilinear")


def _geometry_from_q2() -> tuple[str, str]:
    """Return (strategy, interpolation) of the Q2 winner."""
    import os

    from data import RESULTS_DIR

    path = os.path.join(RESULTS_DIR, "q2_strategies.json")
    if not os.path.exists(path):
        return _FALLBACK_GEOMETRY
    import json

    with open(path) as fh:
        q2 = json.load(fh)
    strategy, _, interp = q2["best_config"].partition("|")
    return strategy, interp


BEST_STRATEGY, BEST_INTERP = _geometry_from_q2()


# ----------------------------------------------------------------------------
# 1. M0 baseline
# ----------------------------------------------------------------------------


def run_m0(seeds=SEEDS, extractor=None) -> dict:
    print("=" * 72)
    print("M0  baseline: frozen ResNet-18 + LogisticRegression, no augmentation")
    print("=" * 72)
    m0 = run_config(
        RunConfig(label="M0", strategy=BEST_STRATEGY, interpolation=BEST_INTERP),
        seeds=seeds, extractor=extractor,
    )
    a = m0
    print(f"  M0  macro_f1 {format_pm(a['macro_f1_mean'], a['macro_f1_std'])}  "
          f"acc {format_pm(a['top1_accuracy_mean'], a['top1_accuracy_std'])}  "
          f"worst {format_pm(a['worst_class_recall_mean'], a['worst_class_recall_std'])} "
          f"(cl{a['worst_class_index']+1:02d})")
    return m0


# ----------------------------------------------------------------------------
# 2. Semantic integrity, measured before training
# ----------------------------------------------------------------------------


def semantic_integrity(spec: AugmentSpec, target_labels, m0, seeds=SEEDS,
                       n_replicates: int = 3, max_samples: int = 120,
                       extractor=None) -> dict:
    """Does the augmentation preserve the class identity of its own source?

    Two independent readouts, both in feature space:

    * **cosine to source** -- similarity between an augmented sample and the
      pristine image it came from. A drop means the sample drifted.
    * **prototype agreement** -- is the augmented sample's nearest class
      centroid (fitted on *clean* training features) still its own class?
      This is the sharper test: drift that stays inside the class is harmless,
      drift that crosses a decision region is not.

    Prototypes are built only from the targeted classes, so the readout is
    conditioned on exactly the classes the augmentation is aimed at.
    """
    import torch
    from torch.utils.data import DataLoader

    from augmentation import AugmentedFewShotDataset, expand_records
    from data import build_transform, make_split
    from models import extract_features

    extractor = extractor or get_feature_extractor()
    set_global_seed(seeds[0])
    split = make_split(seeds[0])
    target = set(target_labels or ())

    records3 = expand_records(split.train, target, n_replicates)[:max_samples]
    ds = AugmentedFewShotDataset(records3, build_transform(BEST_STRATEGY, BEST_INTERP),
                                 spec, target, seeds[0])
    gen = torch.Generator()
    gen.manual_seed(seeds[0])
    X, y, _, _ = extract_features(
        extractor, DataLoader(ds, batch_size=32, shuffle=False, generator=gen),
        cache_key=f"integ_{spec.name}_{seeds[0]}_{len(records3)}")

    Xn = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-8)

    # index of the pristine (replicate 0) copy of each source image
    clean_idx: dict[tuple[str, int], int] = {}
    for i, (p, yy, r) in enumerate(records3):
        if r == 0:
            clean_idx[(p, int(yy))] = i

    # class prototypes from clean samples only
    protos, proto_labels = [], []
    for c in sorted(target):
        idx = [i for i, (p, yy, r) in enumerate(records3) if r == 0 and yy == c]
        if not idx:
            continue
        v = Xn[idx].mean(0)
        protos.append(v / max(float(np.linalg.norm(v)), 1e-8))
        proto_labels.append(c)

    if not protos:
        return {"spec": spec.to_dict(), "n_augmented_checked": 0,
                "interpretation": "no clean samples for the targeted classes; "
                                  "integrity could not be measured"}

    P = np.vstack(protos)                                   # (n_proto, 512)
    nearest = np.asarray(proto_labels)[(P @ Xn.T).argmax(axis=0)]  # (n_samples,)

    cos, agree = [], []
    for i, (p, yy, r) in enumerate(records3):
        if r == 0:
            continue
        j = clean_idx.get((p, int(yy)))
        if j is not None:
            cos.append(float(Xn[i] @ Xn[j]))
        agree.append(float(nearest[i] == int(yy)))

    return {
        "spec": spec.to_dict(),
        "n_augmented_checked": len(cos),
        "cosine_to_source_mean": float(np.mean(cos)) if cos else None,
        "cosine_to_source_p05": float(np.percentile(cos, 5)) if cos else None,
        "cosine_to_source_min": float(np.min(cos)) if cos else None,
        "prototype_agreement": float(np.mean(agree)) if agree else None,
        "interpretation": _integrity_sentence(spec, cos, agree),
    }


def _integrity_sentence(spec: AugmentSpec, cos, agree) -> str:
    if not cos:
        return "no augmented samples were produced, so integrity could not be measured"
    c = float(np.mean(cos))
    a = float(np.mean(agree)) if agree else None
    parts = [f"mean cosine to source {c:.3f}"]
    if a is not None:
        parts.append(f"prototype agreement {a:.3f}")
    verdict = (
        "semantics look preserved" if (c > 0.85 and (a is None or a > 0.9))
        else "borderline: measurable drift" if (c > 0.7 and (a is None or a > 0.75))
        else "semantics likely damaged"
    )
    return f"{spec.name}: " + ", ".join(parts) + f" -> {verdict}"


# ----------------------------------------------------------------------------
# 3. M1 targeted augmentation sweep
# ----------------------------------------------------------------------------


def run_m1_sweep(target_labels, class_names, spec: AugmentSpec, seeds=SEEDS,
                 extractor=None, ratios=RATIO_SWEEP) -> dict:
    """Sweep the per-class augmentation ratio on the targeted classes only."""
    print()
    print("=" * 72)
    print(f"M1  targeted augmentation '{spec.name}' on "
          f"{[class_names[i] for i in sorted(target_labels)]}")
    print("=" * 72)
    out = {}
    for ratio in ratios:
        reps = int(round(ratio)) - 1
        label = f"M1_{spec.name}_x{ratio:g}"
        print(f"  [{label}]  per-class ratio {ratio:g}x on targeted classes")
        out[label] = run_config(
            RunConfig(label=label, strategy=BEST_STRATEGY, interpolation=BEST_INTERP,
                      spec=spec, target_labels=target_labels, n_replicates=reps),
            seeds=seeds, extractor=extractor,
        )
        a = out[label]
        print(f"    -> macro_f1 {format_pm(a['macro_f1_mean'], a['macro_f1_std'])}  "
              f"worst {format_pm(a['worst_class_recall_mean'], a['worst_class_recall_std'])}  "
              f"({a['wall_seconds']}s)\n")
    return out


def turning_point(sweep: dict, m0: dict) -> dict:
    """Locate the ratio where Macro-F1 or worst-class recall stops paying.

    Defined as the first ratio whose Macro-F1 interval no longer overlaps the
    best ratio's interval, and reported alongside the raw sequence so the
    reader can see whether the curve is flat, peaked or monotonic. A sweep with
    no non-overlapping step returns no turning point rather than a fabricated
    one.
    """
    labels = list(sweep)
    f1 = np.array([sweep[k]["macro_f1_mean"] for k in labels])
    sd = np.array([sweep[k]["macro_f1_std"] for k in labels])
    wr = np.array([sweep[k]["worst_class_recall_mean"] for k in labels])
    ratios = np.array([sweep[k]["config"]["n_replicates"] + 1 for k in labels])

    best_i = int(np.argmax(f1))
    tp = None
    for i in range(best_i + 1, len(labels)):
        if not intervals_overlap(f1[best_i], sd[best_i], f1[i], sd[i]):
            tp = {"ratio": float(ratios[i]), "label": labels[i],
                  "reason": f"Macro-F1 at {ratios[i]:g}x no longer overlaps the "
                            f"{ratios[best_i]:g}x interval"}
            break

    improves_over_m0 = intervals_overlap(
        m0["macro_f1_mean"], m0["macro_f1_std"], f1[best_i], sd[best_i])
    return {
        "ratios": ratios.tolist(),
        "macro_f1": f1.tolist(), "macro_f1_std": sd.tolist(),
        "worst_class_recall": wr.tolist(),
        "best_ratio": float(ratios[best_i]),
        "best_macro_f1": float(f1[best_i]),
        "m0_macro_f1": m0["macro_f1_mean"],
        "turning_point": tp,
        "m1_beats_m0_separably": not improves_over_m0,
        "verdict": _turning_point_sentence(ratios, f1, sd, m0, tp, best_i, improves_over_m0),
    }


def _turning_point_sentence(ratios, f1, sd, m0, tp, best_i, overlaps_m0) -> str:
    seq = ", ".join(f"{r:g}x:{v:.3f}" for r, v in zip(ratios, f1))
    base = f"Macro-F1 across ratios -- {seq}. "
    if tp:
        base += f"Turning point at {tp['ratio']:g}x: {tp['reason']}."
    else:
        base += ("No ratio's interval separated from the best, so no turning point is "
                 "claimed; the curve is flat within seed noise over the range tested.")
    if overlaps_m0:
        base += (f" The best ratio's interval still overlaps M0 "
                 f"({m0['macro_f1_mean']:.4f}±{m0['macro_f1_std']:.4f}), so targeted "
                 "augmentation is not shown to help at this sample size.")
    else:
        base += (f" The best ratio's interval is disjoint from M0 "
                 f"({m0['macro_f1_mean']:.4f}±{m0['macro_f1_std']:.4f}).")
    return base


# ----------------------------------------------------------------------------
# 4. Orchestration
# ----------------------------------------------------------------------------


def run(seeds=SEEDS) -> dict:
    set_global_seed(seeds[0])
    ensure_dirs()
    extractor = get_feature_extractor()

    from data import get_class_paths
    class_names = [f"cl{i+1:02d}" for i in range(20)]
    assert all(p for p in get_class_paths()), "dataset discovery failed"

    m0 = run_m0(seeds=seeds, extractor=extractor)
    weak = rank_weak_classes(m0, class_names, top_k=N_TARGET_CLASSES)
    target_labels = {i for _, i, _, _ in weak}
    print()
    print("weak classes (lowest mean recall across the 3 seeds):")
    for name, idx, mu, sd in weak:
        print(f"  {name}  recall {mu:.3f} ± {sd:.3f}")
    print()

    integrity = {name: semantic_integrity(spec, target_labels, m0, seeds=seeds,
                                          extractor=extractor)
                 for name, spec in SPECS.items() if name != "none"}
    print("semantic integrity of the recipes (checked before training):")
    for name, r in integrity.items():
        print(f"  {r['interpretation']}")
    print()

    sweeps = {
        name: run_m1_sweep(target_labels, class_names, spec, seeds=seeds,
                           extractor=extractor)
        for name, spec in SPECS.items() if name in ("mild", "mild_cutout", "strong")
    }
    points = {name: turning_point(sw, m0) for name, sw in sweeps.items()}

    # failure cases: weak classes whose recall never recovers
    best_name = max(points, key=lambda k: points[k]["best_macro_f1"])
    best_sw = sweeps[best_name]
    best_ratio_label = max(best_sw, key=lambda k: best_sw[k]["macro_f1_mean"])
    best = best_sw[best_ratio_label]
    pcr0 = np.asarray(m0["per_class_recall_mean"])
    pcr1 = np.asarray(best["per_class_recall_mean"])
    failed = [
        {"class": class_names[i], "m0_recall": float(pcr0[i]), "m1_recall": float(pcr1[i]),
         "delta": float(pcr1[i] - pcr0[i])}
        for i in sorted(target_labels) if pcr1[i] <= pcr0[i] + 0.05
    ]

    out = {
        "question": "Q3 custom few-shot augmentation (M0 -> M1)",
        "seeds": list(seeds),
        "geometry": {"strategy": BEST_STRATEGY, "interpolation": BEST_INTERP,
                     "note": "carried over from Q2 so Q3 isolates the augmentation effect"},
        "m0": m0,
        "weak_classes": [{"class": n, "index": i, "recall_mean": mu, "recall_std": sd}
                         for n, i, mu, sd in weak],
        "target_labels": sorted(target_labels),
        "semantic_integrity": integrity,
        "sweeps": sweeps,
        "turning_points": points,
        "best_recipe": best_name,
        "best_ratio_label": best_ratio_label,
        "best_m1": best,
        "comparison_best_m1_vs_m0": describe_comparison("M0", m0, "M1", best),
        "failure_cases_unrecovered": failed,
        "conclusion": _conclusion(m0, best, points, best_name, failed),
    }
    save_json(out, "q3_augmentation.json")
    return out


def _conclusion(m0, m1, points, best_name, failed) -> str:
    tp = points[best_name]
    parts = [
        f"M0 Macro-F1 {m0['macro_f1_mean']:.4f}±{m0['macro_f1_std']:.4f}; "
        f"best M1 ('{best_name}' at {tp['best_ratio']:g}x) "
        f"{m1['macro_f1_mean']:.4f}±{m1['macro_f1_std']:.4f}.",
        tp["verdict"],
    ]
    if failed:
        parts.append(
            f"{len(failed)} of the targeted classes did not improve by more than 0.05 recall "
            f"({', '.join(f['class'] for f in failed)}); these are the failures that motivate "
            "the M2 diagnosis rather than a further sweep."
        )
    return " ".join(parts)


def _print_summary(r: dict) -> None:
    print("=" * 72)
    print("Q3 summary")
    print("=" * 72)
    print(f"weak classes targeted: {[w['class'] for w in r['weak_classes']]}")
    for name, p in r["turning_points"].items():
        print(f"  '{name}': best {p['best_ratio']:g}x  macro_f1 {p['best_macro_f1']:.4f}  "
              f"turning point {p['turning_point']['ratio'] if p['turning_point'] else 'not established'}")
    print()
    print(r["comparison_best_m1_vs_m0"])
    print()
    print(r["conclusion"])


if __name__ == "__main__":
    _print_summary(run())
