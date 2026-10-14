"""HW02 stage 2: the four mandatory M1 diagnostics.

Each ``diagN_*`` function runs one diagnostic and returns a JSON-serialisable
dict.  Rendering lives in :mod:`run_hw02`, which is the only module that writes
files, so the diagnostics stay testable in isolation.

The question behind every diagnostic is the same: *why* does M1 still miss the
samples it misses, and can that failure be fixed by a better classifier rather
than by more pixels?  Stage 2 answers it, in order:

1. error samples   -- which confusions are stable across seeds, and what does
   the confusion look like next to the prototype of the class it is mistaken
   for?  Stable, semantically similar pairs point at the classifier, not the
   data.
2. learning curve  -- does the probe keep improving with more per-class
   samples, or has it saturated?  A saturated curve rules out "just add data".
3. train/val gap   -- is the probe under-fit or over-fit?  A large gap says the
   regulariser / feature scale is the lever (motivating Variant A).
4. feature space   -- are the problem classes actually separable in the frozen
   features?  Inter-class prototype cosine distance and intra-class variance
   say whether the head is the bottleneck (motivating Variant B).
"""

from __future__ import annotations

import os
import random
import sys
from collections import defaultdict

import numpy as np

HW02_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HW02_DIR)
HW01_DIR = os.path.join(REPO_ROOT, "HW01")
for _p in (HW01_DIR, HW02_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from hw01.data import SEEDS, make_split  # noqa: E402

from hw02_m2_models import (  # noqa: E402
    CLASS_NAMES, GEOMETRY_INTERP, GEOMETRY_STRATEGY, M1_REPLICATES, M1_SPEC_NAME,
    PROBLEM_CLASSES, TARGET_LABELS, LinearLogisticProbe, aggregate_blocks,
    build_bank, evaluate_m0, evaluate_m1, proba_of,
)

# ----------------------------------------------------------------------------
# 1. Error samples
# ----------------------------------------------------------------------------


def diag1_error_samples(seeds=SEEDS, extractor=None, device: str = "cpu",
                        max_examples_per_seed: int = 40) -> dict:
    """Stable misclassification pairs + a representative sample per pair.

    For every seed the M1 probe is fit on the full M1 bank and run over the
    validation bank.  Misclassifications are grouped by ``(true, predicted)``
    so a pair that recurs across all three seeds can be told apart from a
    one-off draw.  For each surviving pair a representative validation image is
    recorded together with the *most typical* training image of the predicted
    class (the one closest to that class's feature prototype), which is the
    reference a human needs to judge the confusion.
    """
    from hw01.models import get_feature_extractor

    from sklearn.metrics import confusion_matrix

    extractor = extractor or get_feature_extractor(device)
    per_seed = []
    mean_cm = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=float)
    pair_info = defaultdict(lambda: {"seeds": set(), "examples": [], "confs": []})

    for seed in seeds:
        bank_tr = build_bank("M1", seed, extractor, device)
        bank_va = build_bank("val", seed, extractor, device)
        probe = LinearLogisticProbe(C=1.0, seed=seed).fit(bank_tr.X, bank_tr.y)
        yhat = probe.predict(bank_va.X)
        proba = proba_of(probe, bank_va.X)

        cm = confusion_matrix(bank_va.y, yhat, labels=list(range(len(CLASS_NAMES)))).astype(float)
        cm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1e-9)
        mean_cm += cm / len(seeds)

        errors = []
        for i in range(len(bank_va)):
            true, pred = int(bank_va.y[i]), int(yhat[i])
            if true == pred:
                continue
            conf = float(proba[i, pred])
            rec = {"path": bank_va.paths[i], "true": true, "pred": pred,
                   "true_name": CLASS_NAMES[true], "pred_name": CLASS_NAMES[pred],
                   "confidence": conf}
            errors.append(rec)
            key = (true, pred)
            pair_info[key]["seeds"].add(seed)
            pair_info[key]["confs"].append(conf)
            pair_info[key]["examples"].append((seed, rec))
        errors.sort(key=lambda e: -e["confidence"])
        per_seed.append({"seed": seed, "n_errors": len(errors),
                         "errors": errors[:max_examples_per_seed]})

    # prototype of each class, and the training image closest to it, per seed
    proto_lookup = _prototype_lookup(seeds, extractor, device)

    pairs = []
    for (true, pred), info in pair_info.items():
        seed_example = max(info["examples"], key=lambda t: t[1]["confidence"])
        seed, example = seed_example
        proto = proto_lookup.get((seed, pred))
        pairs.append({
            "true": true, "pred": pred,
            "true_name": CLASS_NAMES[true], "pred_name": CLASS_NAMES[pred],
            "n_seeds": len(info["seeds"]), "seeds": sorted(info["seeds"]),
            "n_occurrences": len(info["examples"]),
            "mean_confidence": float(np.mean(info["confs"])),
            "example": {"seed": seed, "path": example["path"],
                        "confidence": example["confidence"]},
            "predicted_class_prototype_train_image": proto,
        })
    pairs.sort(key=lambda p: (-p["n_seeds"], -p["n_occurrences"], -p["mean_confidence"]))

    return {
        "question": "Which samples does M1 still miss, and are those confusions stable?",
        "geometry": {"strategy": GEOMETRY_STRATEGY, "interpolation": GEOMETRY_INTERP},
        "m1_recipe": {"spec": M1_SPEC_NAME, "replicates": M1_REPLICATES,
                      "targets": TARGET_LABELS},
        "n_val_per_seed": 60,
        "mean_normalised_confusion_matrix": mean_cm.tolist(),
        "class_names": CLASS_NAMES,
        "pairs": pairs,
        "per_seed": per_seed,
        "note": ("prototype image = the training image whose frozen feature is closest "
                 "to the mean feature of the predicted class; stable pairs that map "
                 "semantically similar classes point at the classifier, not the pixels"),
    }


def _prototype_lookup(seeds, extractor, device) -> dict:
    """``(seed, class) -> {path, cosine}`` for the most central training image."""
    out = {}
    for seed in seeds:
        bank = build_bank("M0", seed, extractor, device)
        Xn = bank.X / np.maximum(np.linalg.norm(bank.X, axis=1, keepdims=True), 1e-8)
        for c in range(len(CLASS_NAMES)):
            idx = np.where(bank.y == c)[0]
            if idx.size == 0:
                continue
            mu = Xn[idx].mean(axis=0)
            mu = mu / max(float(np.linalg.norm(mu)), 1e-8)
            sims = Xn[idx] @ mu
            j = int(idx[int(np.argmax(sims))])
            out[(seed, c)] = {"path": bank.paths[j], "cosine": float(sims.max())}
    return out


# ----------------------------------------------------------------------------
# 2. Learning curve
# ----------------------------------------------------------------------------


#: Per-class training caps.  ``None`` means "all".
K_VALUES = (5, 10, 15, None)


def _subsample_base(split, k: int | None, seed: int) -> list[int]:
    """Draw up to ``k`` *original* images per class, without replacement."""
    per_class = defaultdict(list)
    for i, (_, y) in enumerate(split.train):
        per_class[int(y)].append(i)
    rng = random.Random(seed * 1000 + (10 ** 6 if k is None else k))
    chosen: list[int] = []
    for c in sorted(per_class):
        idxs = per_class[c]
        take = len(idxs) if k is None else min(k, len(idxs))
        chosen.extend(rng.sample(idxs, take))
    return sorted(chosen)


def diag2_learning_curve(seeds=SEEDS, extractor=None, device: str = "cpu",
                         k_values=K_VALUES) -> dict:
    """Macro-F1 / Top-1 of M0 and M1 as the per-class training cap grows.

    The cap applies to the *original* images per class; M1 then applies its
    ``mild`` 2x recipe to the (already capped) target classes.  That isolates
    "does more data help?" from "does augmentation help?", which a cap on the
    total augmented count would confound.
    """
    from hw01.models import get_feature_extractor

    extractor = extractor or get_feature_extractor(device)
    curve = {m: {"macro_f1_mean": [], "macro_f1_std": [], "top1_accuracy_mean": [],
                 "top1_accuracy_std": [], "n_train_effective": []} for m in ("M0", "M1")}
    per_seed = {m: {} for m in ("M0", "M1")}

    for k in k_values:
        blocks = {m: [] for m in ("M0", "M1")}
        for seed in seeds:
            split = make_split(seed)
            bank_va = build_bank("val", seed, extractor, device)
            chosen = _subsample_base(split, k, seed)
            for m in ("M0", "M1"):
                bank_tr = build_bank(m, seed, extractor, device)
                Xtr, ytr, _, _ = bank_tr.select(chosen)
                probe = LinearLogisticProbe(C=1.0, seed=seed).fit(Xtr, ytr)
                from hw01.metrics import compute_metrics

                met = compute_metrics(bank_va.y, probe.predict(bank_va.X), num_classes=20)
                blocks[m].append({"seed": seed, "macro_f1": met["macro_f1"],
                                  "top1_accuracy": met["top1_accuracy"],
                                  "n_train": int(len(ytr))})
        key = "All" if k is None else str(k)
        for m in ("M0", "M1"):
            f1 = [b["macro_f1"] for b in blocks[m]]
            t1 = [b["top1_accuracy"] for b in blocks[m]]
            ntr = [b["n_train"] for b in blocks[m]]
            curve[m]["macro_f1_mean"].append(float(np.mean(f1)))
            curve[m]["macro_f1_std"].append(float(np.std(f1, ddof=1)))
            curve[m]["top1_accuracy_mean"].append(float(np.mean(t1)))
            curve[m]["top1_accuracy_std"].append(float(np.std(t1, ddof=1)))
            curve[m]["n_train_effective"].append(float(np.mean(ntr)))
            per_seed[m][key] = blocks[m]

    return {
        "question": "Is the linear probe saturated, or would more per-class data still help?",
        "geometry": {"strategy": GEOMETRY_STRATEGY, "interpolation": GEOMETRY_INTERP},
        "k_values": ["All" if k is None else k for k in k_values],
        "sampling": ("without replacement, per class, seeded random.Random(seed*1000+k); "
                     "applied to original images, then M1 augments the capped targets"),
        "curve": curve,
        "per_seed": per_seed,
    }


# ----------------------------------------------------------------------------
# 3. Train / validation gap
# ----------------------------------------------------------------------------


def diag3_train_val_gap(seeds=SEEDS, extractor=None, device: str = "cpu") -> dict:
    """Train vs validation Accuracy, Macro-F1 and cross-entropy for M0 and M1."""
    from hw01.models import get_feature_extractor

    extractor = extractor or get_feature_extractor(device)
    out = {}
    for m, fn in (("M0", evaluate_m0), ("M1", evaluate_m1)):
        blocks = [fn(seed, extractor, device) for seed in seeds]
        agg = aggregate_blocks(blocks)
        out[m] = {
            "train_top1_accuracy": _pm(agg, "train_top1_accuracy"),
            "val_top1_accuracy": _pm(agg, "top1_accuracy"),
            "train_macro_f1": _pm(agg, "train_macro_f1"),
            "val_macro_f1": _pm(agg, "macro_f1"),
            "train_log_loss": _pm(agg, "train_log_loss"),
            "val_log_loss": _pm(agg, "val_log_loss"),
            "gaps": {"top1": round(agg["top1_gap_mean"], 4),
                     "macro_f1": round(agg["macro_f1_gap_mean"], 4),
                     "log_loss": round(agg["loss_gap_mean"], 4)},
            "per_seed": [{"seed": b["seed"],
                          "train_acc": b["train"]["top1_accuracy"],
                          "val_acc": b["val"]["top1_accuracy"],
                          "train_f1": b["train"]["macro_f1"],
                          "val_f1": b["val"]["macro_f1"],
                          "train_loss": b["train_log_loss"],
                          "val_loss": b["val_log_loss"]} for b in blocks],
        }
    out["question"] = "Is M1 under-fit or over-fit?  Which side sets the ceiling?"
    out["note"] = ("gap = validation - train.  A *negative* accuracy gap (train above "
                   "validation) together with a positive loss gap is the over-fitting "
                   "signature: the probe reaches zero training error on 7-21 images per "
                   "class but does not transfer all of it, which is where the regulariser "
                   "and the feature scale become the levers (Variant A)")
    return out


def _pm(agg: dict, key: str) -> dict:
    return {"mean": float(agg[f"{key}_mean"]), "std": float(agg[f"{key}_std"])}


# ----------------------------------------------------------------------------
# 4. Feature-space geometry (t-SNE / UMAP + prototype distances)
# ----------------------------------------------------------------------------


def diag4_feature_space(seed: int = SEEDS[0], extractor=None, device: str = "cpu",
                        perplexity: float = 30.0) -> dict:
    """2-D embedding of the frozen features + prototype-distance statistics.

    One representative seed is embedded (t-SNE is a qualitative view; the
    quantitative readouts -- inter-class prototype cosine distance and
    intra-class variance -- are computed for all 20 classes and are
    seed-stable).  Categories distinguish pristine training images, M1
    augmented images, and validation images, splitting the validation points
    into correct and misclassified by the M1 probe.
    """
    from hw01.models import get_feature_extractor

    extractor = extractor or get_feature_extractor(device)
    bank0 = build_bank("M0", seed, extractor, device)
    bank1 = build_bank("M1", seed, extractor, device)
    bankv = build_bank("val", seed, extractor, device)

    # M1 probe -> which validation points are wrong
    probe = LinearLogisticProbe(C=1.0, seed=seed).fit(bank1.X, bank1.y)
    yhat_val = probe.predict(bankv.X)

    # original train rows, augmented-only rows, then validation rows
    aug_mask = np.asarray([r > 0 for _, _, r in bank1.records3])
    X = np.vstack([bank0.X, bank1.X[aug_mask], bankv.X]).astype(np.float32)
    labels = np.concatenate([bank0.y, bank1.y[aug_mask], bankv.y]).astype(int)
    category = (["original_train"] * len(bank0)
                + ["augmented_train"] * int(aug_mask.sum())
                + ["val_correct"] * len(bankv))
    paths = list(bank0.paths) + list(np.asarray(bank1.paths, dtype=object)[aug_mask]) + list(bankv.paths)
    correct = np.concatenate([np.ones(len(bank0), bool), np.ones(int(aug_mask.sum()), bool),
                              (yhat_val == bankv.y)])
    for i in range(len(bankv)):
        if category[len(bank0) + int(aug_mask.sum()) + i] == "val_correct" and not correct[len(bank0) + int(aug_mask.sum()) + i]:
            category[len(bank0) + int(aug_mask.sum()) + i] = "val_wrong"

    Xn = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-8)

    tsne_xy = _tsne(Xn, perplexity=perplexity, seed=seed)
    umap_xy = _umap(Xn, seed=seed)

    # prototype statistics from pristine training features
    P, proto_names = _prototypes(Xn[:len(bank0)], bank0.y)
    cos = np.clip(P @ P.T, -1.0, 1.0)
    dist = 1.0 - cos
    var = _intra_class_variance(Xn[:len(bank0)], bank0.y, P, proto_names)

    problem_names = [CLASS_NAMES[c] for c in PROBLEM_CLASSES]
    prob_idx = [proto_names.index(n) for n in problem_names if n in proto_names]

    return {
        "question": "Are the problem classes separable in the frozen features?",
        "seed": seed,
        "geometry": {"strategy": GEOMETRY_STRATEGY, "interpolation": GEOMETRY_INTERP},
        "preprocessing": "L2-normalised 512-d features",
        "embedding_methods": ["tsne"] + (["umap"] if umap_xy is not None else []),
        "points": {
            "label": labels.tolist(),
            "label_name": [CLASS_NAMES[c] for c in labels],
            "category": category,
            "is_problem_class": [int(c) in PROBLEM_CLASSES for c in labels],
            "tsne": tsne_xy.tolist(),
            "umap": umap_xy.tolist() if umap_xy is not None else None,
            "val_paths_wrong": [bankv.paths[i] for i in range(len(bankv))
                                if yhat_val[i] != bankv.y[i]],
        },
        "prototype_cosine_distance": {
            "labels": [proto_names[i] for i in prob_idx],
            "matrix": dist[np.ix_(prob_idx, prob_idx)].tolist(),
        },
        "prototype_cosine_distance_all": {
            "labels": proto_names,
            "matrix": dist.tolist(),
        },
        "intra_class_variance": var,
        "problem_classes": problem_names,
        "note": ("distance = 1 - cosine between class prototypes; low distance between "
                 "two classes is the geometric signature of the stable confusions in "
                 "diagnostic 1, and bounds what any linear head can do"),
    }


def _prototypes(Xn: np.ndarray, y: np.ndarray):
    names, protos = [], []
    for c in range(len(CLASS_NAMES)):
        idx = np.where(y == c)[0]
        if idx.size == 0:
            continue
        mu = Xn[idx].mean(axis=0)
        mu = mu / max(float(np.linalg.norm(mu)), 1e-8)
        protos.append(mu)
        names.append(CLASS_NAMES[c])
    return np.vstack(protos).astype(np.float32), names


def _intra_class_variance(Xn: np.ndarray, y: np.ndarray, P: np.ndarray, names) -> dict:
    out = {}
    for c in range(len(CLASS_NAMES)):
        idx = np.where(y == c)[0]
        if idx.size == 0:
            continue
        mu = P[names.index(CLASS_NAMES[c])]
        cos = np.clip(Xn[idx] @ mu, -1.0, 1.0)
        out[CLASS_NAMES[c]] = {
            "n_train": int(idx.size),
            "mean_radial_distance": float(np.mean(np.linalg.norm(Xn[idx] - mu, axis=1))),
            "mean_cosine_to_prototype": float(np.mean(cos)),
            "circular_variance": float(1.0 - np.mean(cos)),
        }
    return out


def _tsne(Xn: np.ndarray, perplexity: float, seed: int) -> np.ndarray:
    from sklearn.manifold import TSNE

    perp = float(min(perplexity, max(5.0, (Xn.shape[0] - 1) / 3.0)))
    model = TSNE(n_components=2, perplexity=perp, init="pca",
                 learning_rate="auto", random_state=seed, max_iter=1000)
    return model.fit_transform(Xn)


def _umap(Xn: np.ndarray, seed: int):
    try:
        import umap  # type: ignore
    except Exception:
        return None
    try:
        reducer = umap.UMAP(n_components=2, random_state=seed, n_neighbors=15, min_dist=0.1)
        return reducer.fit_transform(Xn)
    except Exception:
        return None
