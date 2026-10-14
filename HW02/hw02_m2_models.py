"""HW02 stage 2: M2 candidate models and the shared feature bank.

Stage 1 fixed the backbone (frozen ImageNet ResNet-18) and the classifier
family (``sklearn`` LogisticRegression) and delivered M0 (no augmentation) and
M1 (targeted ``mild`` 2x augmentation on the five weak classes).  This module
adds what the second submission needs:

* a **feature bank** -- one non-shuffled forward pass per (seed, model) so every
  diagnostic can index rows back to the exact source image.  A shuffled bank
  cannot support this, because a targeted class contributes several rows that
  share the same path (its replicates) and become indistinguishable once the
  loader shuffles them.
* three **M2 candidates**, each changing a different half of the pipeline:
  - Variant A  re-scale the *features* (L2 normalisation) and re-tune the
    regulariser / class weighting of the same linear probe;
  - Variant B  replace the *classifier* (cosine prototype, or a linear SVM);
  - Variant C  augment in *feature space* instead of pixel space.
* the aggregate comparison of M0, M1 and every M2 candidate, phrased only by
  whether the +/-1 sd intervals overlap (three seeds cannot support a
  significance claim).

Geometry (``direct + bilinear``) and the M1 recipe (``mild``, 2x on five weak
classes) are read from the stage-1 result files, never hardcoded, so stage 2
cannot silently drift from what stage 1 reported.
"""

from __future__ import annotations

import json
import os
import random
import sys
from dataclasses import dataclass, field

import numpy as np

# ----------------------------------------------------------------------------
# Bootstrap: HW02 sits beside HW01 and reuses the hw01 library + feature cache.
# ----------------------------------------------------------------------------

HW02_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HW02_DIR)
HW01_DIR = os.path.join(REPO_ROOT, "HW01")
for _p in (HW01_DIR, HW02_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import log_loss  # noqa: E402
from sklearn.svm import LinearSVC  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from hw01.augmentation import SPECS, AugmentedFewShotDataset  # noqa: E402
from hw01.data import (  # noqa: E402
    SEEDS, build_transform, cache_key, make_split,
)
from hw01.metrics import aggregate, compute_metrics, intervals_overlap  # noqa: E402
from hw01.models import extract_features, get_feature_extractor  # noqa: E402

# ----------------------------------------------------------------------------
# Output locations (HW02 owns its own results/ and plots/)
# ----------------------------------------------------------------------------

HW02_RESULTS_DIR = os.path.join(HW02_DIR, "results")
HW02_FULL_RESULTS_DIR = os.path.join(HW02_RESULTS_DIR, "full")
HW02_PLOTS_DIR = os.path.join(HW02_DIR, "plots")
_HW01_FULL_RESULTS = os.path.join(HW01_DIR, "results", "full")


def ensure_hw02_dirs() -> None:
    for d in (HW02_RESULTS_DIR, HW02_FULL_RESULTS_DIR, HW02_PLOTS_DIR):
        os.makedirs(d, exist_ok=True)


def save_json(obj, filename: str, full: bool = True) -> str:
    """Write a JSON artefact to HW02's own results tree."""
    ensure_hw02_dirs()
    path = os.path.join(HW02_FULL_RESULTS_DIR if full else HW02_RESULTS_DIR, filename)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, default=str)
    return path


# ----------------------------------------------------------------------------
# Stage-1 provenance (geometry, M1 recipe, targets)
# ----------------------------------------------------------------------------


def _read_stage1(filename: str) -> dict | None:
    path = os.path.join(_HW01_FULL_RESULTS, filename)
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def geometry_from_stage1() -> tuple[str, str]:
    """Q2 winner; falls back to the assignment's direct + bilinear baseline."""
    q2 = _read_stage1("q2_strategies.json") or {}
    best = q2.get("best_config", "direct|bilinear")
    strategy, _, interp = best.partition("|")
    return strategy or "direct", interp or "bilinear"


def targets_from_stage1() -> list[int]:
    """The five classes M1 augmented; falls back to Q3's known set."""
    q3 = _read_stage1("q3_augmentation.json")
    if q3 and q3.get("target_labels"):
        return [int(i) for i in q3["target_labels"]]
    return [0, 5, 6, 11, 17]


GEOMETRY_STRATEGY, GEOMETRY_INTERP = geometry_from_stage1()
M1_SPEC_NAME = "mild"
M1_REPLICATES = 1  # 2x on the targeted classes
TARGET_LABELS = targets_from_stage1()

CLASS_NAMES = [f"cl{i:02d}" for i in range(1, 21)]

#: Classes the diagnostics zoom into (HW02 assignment): cl18, cl10, cl01, cl02, cl12.
PROBLEM_CLASSES = [17, 9, 0, 1, 11]


def get_device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------------------------------------------------------
# Feature bank
# ----------------------------------------------------------------------------


@dataclass
class FeatureBank:
    """Frozen-feature matrix whose rows stay aligned to their source record.

    ``src[i]`` is the index of the *original* ``(path, label)`` entry that row
    ``i`` was derived from.  Replicate rows of a targeted class share the same
    ``src`` value as their pristine original, which is what makes fair
    per-class subsampling possible in the learning-curve diagnostic.
    """

    kind: str
    seed: int
    X: np.ndarray
    y: np.ndarray
    paths: list[str]
    groups: np.ndarray
    records3: list[tuple[str, int, int]]
    src: np.ndarray
    base: list[tuple[str, int]] = field(default_factory=list)

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def select(self, base_indices) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
        """Rows whose source base index is in ``base_indices``."""
        keep = np.isin(self.src, np.asarray(sorted(base_indices)))
        return self.X[keep], self.y[keep], [p for p, k in zip(self.paths, keep) if k], self.groups[keep]


def _expand_base(base: list[tuple[str, int]], targets: set[int], reps: int):
    """Build the ``(path, label, replicate)`` list and its source-index map."""
    records3: list[tuple[str, int, int]] = []
    src: list[int] = []
    for bi, (path, y) in enumerate(base):
        y = int(y)
        records3.append((path, y, 0))
        src.append(bi)
        if reps > 0 and y in targets:
            for r in range(1, reps + 1):
                records3.append((path, y, r))
                src.append(bi)
    return records3, np.asarray(src, dtype=int)


def build_bank(kind: str, seed: int, extractor, device: str = "cpu",
               split=None) -> FeatureBank:
    """Extract a non-shuffled feature bank for one (model kind, seed).

    ``kind`` is ``"M0"`` (pristine train), ``"M1"`` (mild 2x train) or
    ``"val"`` (held-out validation, never augmented).
    """
    split = split or make_split(seed)
    if kind == "val":
        base, spec, targets, reps = split.val, SPECS["none"], set(), 0
    elif kind == "M0":
        base, spec, targets, reps = split.train, SPECS["none"], set(), 0
    elif kind == "M1":
        base, spec, targets, reps = (split.train, SPECS[M1_SPEC_NAME],
                                     set(TARGET_LABELS), M1_REPLICATES)
    else:
        raise ValueError(f"unknown bank kind {kind!r}")

    records3, src = _expand_base(base, targets, reps)
    transform = build_transform(GEOMETRY_STRATEGY, GEOMETRY_INTERP)
    ds = AugmentedFewShotDataset(records3, transform, spec, targets, seed)
    loader = DataLoader(ds, batch_size=64, shuffle=False)
    key = cache_key(stage="hw02", kind=kind, strategy=GEOMETRY_STRATEGY,
                    interpolation=GEOMETRY_INTERP, spec=spec.name, reps=reps,
                    targets=sorted(targets), seed=seed, n=len(records3),
                    device=("cpu" if str(device).startswith("cpu") else "cuda"))
    X, y, g, paths = extract_features(extractor, loader, device=device,
                                      cache_key=f"hw02_{key}")
    return FeatureBank(kind=kind, seed=seed, X=X, y=np.asarray(y), paths=list(paths),
                       groups=np.asarray(g), records3=records3, src=src, base=list(base))


# ----------------------------------------------------------------------------
# Probability helpers (classes may be missing on a tiny training subset)
# ----------------------------------------------------------------------------


def _expand_proba(proba: np.ndarray, classes, num_classes: int = 20) -> np.ndarray:
    out = np.zeros((proba.shape[0], num_classes), dtype=float)
    for j, c in enumerate(np.asarray(classes)):
        c = int(c)
        if 0 <= c < num_classes:
            out[:, c] = proba[:, j]
    rows = out.sum(axis=1, keepdims=True)
    rows[rows == 0] = 1.0
    return out / rows


def proba_of(probe, X: np.ndarray, num_classes: int = 20) -> np.ndarray:
    """Probabilities over the full label set, whatever the probe exposes.

    ``predict_proba`` is used when present.  A margin-only classifier (the
    linear SVM) gets a softmax over its decision values; that is a monotone
    score, not a calibrated probability, and is only ever used for the
    log-loss readout, which is labelled accordingly.
    """
    classes = getattr(probe, "classes_", None)
    if hasattr(probe, "predict_proba"):
        p = np.asarray(probe.predict_proba(X), dtype=float)
        if classes is None:
            classes = np.arange(p.shape[1])
        return _expand_proba(p, classes, num_classes)

    d = np.asarray(probe.decision_function(X), dtype=float)
    if d.ndim == 1:
        d = np.stack([-d, d], axis=1)
    d = d - d.max(axis=1, keepdims=True)
    e = np.exp(d)
    if classes is None:
        classes = np.arange(d.shape[1])
    return _expand_proba(e / e.sum(axis=1, keepdims=True), classes, num_classes)


# ----------------------------------------------------------------------------
# Probes
# ----------------------------------------------------------------------------


def _l2(X: np.ndarray) -> np.ndarray:
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-8)


class LinearLogisticProbe:
    """The stage-1 probe, re-declared here so M0/M1 share one code path.

    Same hyper-parameters as :class:`hw01.models.LogisticRegressionProbe`
    (lbfgs, ``max_iter=5000``, ``random_state=seed``) so the M0/M1 numbers
    reproduce stage 1 exactly; only ``predict_proba`` is added for log-loss.
    """

    name = "linear_lr"

    def __init__(self, C: float = 1.0, seed: int = 0, max_iter: int = 5000):
        self.clf = LogisticRegression(C=C, solver="lbfgs", max_iter=max_iter,
                                      random_state=seed)

    def fit(self, X, y):
        self.clf.fit(X, y)
        self.classes_ = self.clf.classes_
        return self

    def predict(self, X):
        return self.clf.predict(X)

    def predict_proba(self, X):
        return self.clf.predict_proba(X)

    def decision_function(self, X):
        return self.clf.decision_function(X)


class L2NormLogisticProbe:
    """Variant A: L2-normalise each feature vector, then linear probe."""

    name = "l2_lr"

    def __init__(self, C: float = 1.0, class_weight=None, seed: int = 0,
                 max_iter: int = 5000):
        self.clf = LogisticRegression(C=C, solver="lbfgs", max_iter=max_iter,
                                      class_weight=class_weight, random_state=seed)

    def fit(self, X, y):
        self.clf.fit(_l2(X), y)
        self.classes_ = self.clf.classes_
        return self

    def predict(self, X):
        return self.clf.predict(_l2(X))

    def predict_proba(self, X):
        return self.clf.predict_proba(_l2(X))

    def decision_function(self, X):
        return self.clf.decision_function(_l2(X))


class CosinePrototypeClassifier:
    """Variant B: nearest class prototype under cosine similarity.

    No training beyond the class means.  ``tau`` sharpens the softmax used for
    the probability readout; the arg-max label (and therefore every accuracy
    metric) is independent of ``tau``.
    """

    name = "cosine_prototype"

    def __init__(self, tau: float = 10.0):
        self.tau = float(tau)

    def fit(self, X, y):
        Xn = _l2(X)
        self.classes_ = np.unique(y)
        protos = []
        for c in self.classes_:
            v = Xn[y == c].mean(axis=0)
            protos.append(v / max(float(np.linalg.norm(v)), 1e-8))
        self.P_ = np.vstack(protos).astype(np.float32)
        return self

    def decision_function(self, X):
        return self.tau * (_l2(X) @ self.P_.T)

    def predict(self, X):
        return self.classes_[self.decision_function(X).argmax(axis=1)]

    def predict_proba(self, X):
        d = self.decision_function(X)
        d = d - d.max(axis=1, keepdims=True)
        e = np.exp(d)
        return e / e.sum(axis=1, keepdims=True)


class LinearSVMProbe:
    """Variant B (secondary): one-vs-rest linear SVM."""

    name = "linear_svm"

    def __init__(self, C: float = 1.0, seed: int = 0, max_iter: int = 20000):
        self.clf = LinearSVC(C=C, dual="auto", max_iter=max_iter, random_state=seed)

    def fit(self, X, y):
        self.clf.fit(X, y)
        self.classes_ = self.clf.classes_
        return self

    def predict(self, X):
        return self.clf.predict(X)

    def decision_function(self, X):
        return self.clf.decision_function(X)


# ----------------------------------------------------------------------------
# Evaluation + aggregation
# ----------------------------------------------------------------------------


def evaluate_probe(probe, Xtr, ytr, Xva, yva, num_classes: int = 20) -> dict:
    """Fit a probe and return train/val metrics plus the train-val loss gap."""
    probe.fit(Xtr, ytr)
    ytr_hat = probe.predict(Xtr)
    yva_hat = probe.predict(Xva)
    mtr = compute_metrics(ytr, ytr_hat, num_classes)
    mva = compute_metrics(yva, yva_hat, num_classes)
    ptr = proba_of(probe, Xtr, num_classes)
    pva = proba_of(probe, Xva, num_classes)
    labels = list(range(num_classes))
    return {
        "train": mtr,
        "val": mva,
        "train_log_loss": float(log_loss(ytr, ptr, labels=labels)),
        "val_log_loss": float(log_loss(yva, pva, labels=labels)),
    }


def aggregate_blocks(blocks: list[dict]) -> dict:
    """Collapse per-seed :func:`evaluate_probe` outputs into mean +/- sd.

    The validation side goes through :func:`hw01.metrics.aggregate` so the
    schema matches stage 1 (``*_mean``/``*_std``, ``per_class_recall_mean``,
    ``mean_normalised_confusion_matrix``).  Train-side scores and both log
    losses are averaged here.
    """
    val_runs = []
    for b in blocks:
        r = dict(b["val"])
        r["seed"] = b.get("seed")
        val_runs.append(r)
    agg = aggregate(val_runs)

    def ms(key):
        vals = [b[key] for b in blocks]
        return float(np.mean(vals)), float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0

    for name, key in (("train_macro_f1", lambda b: b["train"]["macro_f1"]),
                      ("train_top1_accuracy", lambda b: b["train"]["top1_accuracy"]),
                      ("train_balanced_accuracy", lambda b: b["train"]["balanced_accuracy"]),
                      ("train_worst_class_recall", lambda b: b["train"]["worst_class_recall"]),
                      ("val_log_loss", lambda b: b["val_log_loss"]),
                      ("train_log_loss", lambda b: b["train_log_loss"])):
        vals = [key(b) for b in blocks]
        agg[f"{name}_mean"] = float(np.mean(vals))
        agg[f"{name}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0

    agg["top1_gap_mean"] = agg["top1_accuracy_mean"] - agg["train_top1_accuracy_mean"]
    agg["macro_f1_gap_mean"] = agg["macro_f1_mean"] - agg["train_macro_f1_mean"]
    agg["loss_gap_mean"] = agg["val_log_loss_mean"] - agg["train_log_loss_mean"]
    return agg


# ----------------------------------------------------------------------------
# M0 / M1 (recomputed from the same banks so diagnostics and comparison agree)
# ----------------------------------------------------------------------------


def evaluate_linear(seed: int, kind: str, extractor, device: str = "cpu") -> dict:
    bank_tr = build_bank(kind, seed, extractor, device)
    bank_va = build_bank("val", seed, extractor, device)
    probe = LinearLogisticProbe(C=1.0, seed=seed)
    block = evaluate_probe(probe, bank_tr.X, bank_tr.y, bank_va.X, bank_va.y)
    block["seed"] = seed
    return block


def evaluate_m0(seed, extractor, device="cpu"):
    return evaluate_linear(seed, "M0", extractor, device)


def evaluate_m1(seed, extractor, device="cpu"):
    return evaluate_linear(seed, "M1", extractor, device)


# ----------------------------------------------------------------------------
# Variant A: L2 normalisation + regularisation / class-weight grid
# ----------------------------------------------------------------------------

A_C_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)
A_CLASS_WEIGHTS = (None, "balanced")


def variant_a_configs() -> list[tuple[str, float, object]]:
    out = []
    for C in A_C_GRID:
        for cw in A_CLASS_WEIGHTS:
            out.append((f"C={C:g}|cw={cw or 'None'}", C, cw))
    return out


def evaluate_variant_a(seed: int, extractor, device: str = "cpu") -> dict:
    bank_tr = build_bank("M1", seed, extractor, device)
    bank_va = build_bank("val", seed, extractor, device)
    out = {}
    for label, C, cw in variant_a_configs():
        probe = L2NormLogisticProbe(C=C, class_weight=cw, seed=seed)
        blk = evaluate_probe(probe, bank_tr.X, bank_tr.y, bank_va.X, bank_va.y)
        blk["seed"] = seed
        out[label] = blk
    return out


# ----------------------------------------------------------------------------
# Variant B: alternative classifiers
# ----------------------------------------------------------------------------

B_TAU_GRID = (5.0, 10.0, 20.0, 40.0)
B_SVM_C_GRID = (0.01, 0.1, 1.0, 10.0)


def variant_b_configs() -> list[tuple[str, callable]]:
    out = [(f"cosine_tau{tau:g}", (lambda tau=tau: CosinePrototypeClassifier(tau=tau)))
           for tau in B_TAU_GRID]
    out += [(f"svm_C{C:g}", (lambda C=C: LinearSVMProbe(C=C)))
            for C in B_SVM_C_GRID]
    return out


def evaluate_variant_b(seed: int, extractor, device: str = "cpu") -> dict:
    bank_tr = build_bank("M1", seed, extractor, device)
    bank_va = build_bank("val", seed, extractor, device)
    out = {}
    for label, factory in variant_b_configs():
        blk = evaluate_probe(factory(), bank_tr.X, bank_tr.y, bank_va.X, bank_va.y)
        blk["seed"] = seed
        out[label] = blk
    return out


# ----------------------------------------------------------------------------
# Variant C: feature-space augmentation
# ----------------------------------------------------------------------------


def feature_mixup(X: np.ndarray, y: np.ndarray, targets, rng: np.random.Generator,
                  n_copies: int = 2, max_weight: float = 0.4) -> tuple[np.ndarray, np.ndarray]:
    """Convex combinations towards the class prototype (a feature-space Mixup).

    ``x' = (1 - w) x + w * mu_c`` with ``w ~ U(0, max_weight)``.  Small
    ``w`` keeps the sample on its own class manifold while nudging it towards
    the centroid, which is exactly the direction that reduces intra-class
    variance without leaving the class.
    """
    proto = {c: X[y == c].mean(axis=0) for c in set(int(v) for v in targets)}
    xs, ys = [], []
    for c, mu in proto.items():
        for i in np.where(y == c)[0]:
            for _ in range(n_copies):
                w = float(rng.uniform(0.0, max_weight))
                xs.append((1.0 - w) * X[i] + w * mu)
                ys.append(c)
    if not xs:
        return X, y
    return np.vstack([X, np.asarray(xs, dtype=X.dtype)]), np.concatenate([y, np.asarray(ys)])


def directional_gaussian_noise(X: np.ndarray, y: np.ndarray, targets, rng: np.random.Generator,
                               n_copies: int = 2, sigma: float = 0.3,
                               isotropic: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Perturb towards the prototype plus a small isotropic Gaussian term.

    ``x' = x + eta (mu_c - x) + N(0, isotropic * ||mu_c - x||)`` with
    ``eta ~ U(0, sigma)``.  The Greek step is *directional* (it points at the
    class centroid); the Gaussian term stops every copy from collapsing onto
    the same ray.
    """
    proto = {c: X[y == c].mean(axis=0) for c in set(int(v) for v in targets)}
    xs, ys = [], []
    for c, mu in proto.items():
        for i in np.where(y == c)[0]:
            d = mu - X[i]
            scale = max(float(np.linalg.norm(d)), 1e-8)
            for _ in range(n_copies):
                eta = float(rng.uniform(0.0, sigma))
                noise = rng.normal(0.0, isotropic, size=X.shape[1]).astype(X.dtype) * scale
                xs.append(X[i] + eta * d + noise)
                ys.append(c)
    if not xs:
        return X, y
    return np.vstack([X, np.asarray(xs, dtype=X.dtype)]), np.concatenate([y, np.asarray(ys)])


C_METHODS = {
    "mixup_proto": dict(fn="mixup", n_copies=2, max_weight=0.4),
    "dir_noise": dict(fn="dir_noise", n_copies=2, sigma=0.3, isotropic=0.05),
}


def _augment_features(method: str, X: np.ndarray, y: np.ndarray, seed: int
                      ) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed * 7919 + 13)
    cfg = C_METHODS[method]
    if cfg["fn"] == "mixup":
        return feature_mixup(X, y, TARGET_LABELS, rng, n_copies=cfg["n_copies"],
                             max_weight=cfg["max_weight"])
    return directional_gaussian_noise(X, y, TARGET_LABELS, rng, n_copies=cfg["n_copies"],
                                      sigma=cfg["sigma"], isotropic=cfg["isotropic"])


def evaluate_variant_c(seed: int, extractor, device: str = "cpu",
                       C: float = 1.0) -> dict:
    """Feature-space augmentation on top of the M1 training features."""
    bank_tr = build_bank("M1", seed, extractor, device)
    bank_va = build_bank("val", seed, extractor, device)
    out = {}
    for method in C_METHODS:
        Xaug, yaug = _augment_features(method, bank_tr.X, bank_tr.y, seed)
        probe = LinearLogisticProbe(C=C, seed=seed)
        blk = evaluate_probe(probe, Xaug, yaug, bank_va.X, bank_va.y)
        blk["seed"] = seed
        blk["n_train_effective"] = int(len(yaug))
        out[method] = blk
    return out


def evaluate_variant_c_l2(seed: int, extractor, device: str = "cpu",
                          C: float = 1.0, class_weight=None) -> dict:
    """Variant C combined with Variant A: L2-normalised, feature-augmented probe.

    Kept because it exposes the same C-sensitivity seen in Variant A: at the
    default ``C=1`` the L2-normalised probe is badly under-fit; it only
    recovers once ``C`` is raised to match the unit feature scale.
    """
    bank_tr = build_bank("M1", seed, extractor, device)
    bank_va = build_bank("val", seed, extractor, device)
    out = {}
    for method in C_METHODS:
        Xaug, yaug = _augment_features(method, bank_tr.X, bank_tr.y, seed)
        probe = L2NormLogisticProbe(C=C, class_weight=class_weight, seed=seed)
        blk = evaluate_probe(probe, Xaug, yaug, bank_va.X, bank_va.y)
        blk["seed"] = seed
        blk["n_train_effective"] = int(len(yaug))
        out[method] = blk
    return out


# ----------------------------------------------------------------------------
# Cross-seed aggregation of a per-seed config dictionary
# ----------------------------------------------------------------------------


def aggregate_configs(per_seed_configs: list[dict]) -> dict:
    """``[{config: block, ...}]`` over seeds -> ``{config: aggregated, ...}``."""
    keys = list(per_seed_configs[0])
    return {k: aggregate_blocks([d[k] for d in per_seed_configs]) for k in keys}


def best_config(agg: dict) -> str:
    """Pick the most *stable* config among those indistinguishable from the best.

    The best mean Macro-F1 sets a reference interval.  Any config whose +/-1 sd
    interval overlaps it is indistinguishable with three seeds, so among that
    set the lowest Macro-F1 std wins (tie-break: higher worst-class recall, then
    higher mean, then the label's lexical order).  Choosing the low-variance
    member of an indistinguishable set is the honest reading of "most stable":
    with three seeds, a nominally higher mean is not evidence enough to prefer a
    config that swings more from one split to the next.
    """
    ref = max(agg, key=lambda k: agg[k]["macro_f1_mean"])
    rm, rs = agg[ref]["macro_f1_mean"], agg[ref]["macro_f1_std"]
    eligible = [k for k in agg
                if intervals_overlap(rm, rs, agg[k]["macro_f1_mean"], agg[k]["macro_f1_std"])]

    def rank(k):
        a = agg[k]
        return (-round(a["macro_f1_std"], 6),
                round(a["worst_class_recall_mean"], 6),
                round(a["macro_f1_mean"], 6),
                k)

    return max(eligible, key=rank)


# ----------------------------------------------------------------------------
# Final M0 / M1 / M2 comparison
# ----------------------------------------------------------------------------


def compare_models(models: dict[str, dict]) -> dict:
    """Pairwise interval-overlap verdicts against M0 and M1, plus an md table."""
    def ov(a, b):
        return intervals_overlap(models[a]["macro_f1_mean"], models[a]["macro_f1_std"],
                                 models[b]["macro_f1_mean"], models[b]["macro_f1_std"])

    comparisons = {}
    for m in models:
        for ref in ("M0", "M1"):
            if m == ref:
                continue
            comparisons[f"{m}_vs_{ref}"] = {
                "delta_macro_f1": round(models[m]["macro_f1_mean"]
                                        - models[ref]["macro_f1_mean"], 4),
                "delta_worst_recall": round(models[m]["worst_class_recall_mean"]
                                            - models[ref]["worst_class_recall_mean"], 4),
                "intervals_overlap": ov(m, ref),
            }

    rows = []
    for name, a in models.items():
        rows.append([
            name,
            f"{a['top1_accuracy_mean']:.4f} \u00b1 {a['top1_accuracy_std']:.4f}",
            f"{a['macro_f1_mean']:.4f} \u00b1 {a['macro_f1_std']:.4f}",
            f"{a['worst_class_recall_mean']:.4f}",
            f"{a['loss_gap_mean']:.4f}",
        ])
    table = _md_table(["model", "Top-1", "Macro-F1", "worst-class recall",
                       "train-val loss gap"], rows)
    return {"comparisons": comparisons, "table": table}


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# Orchestration helper used by run_hw02.py
# ----------------------------------------------------------------------------


def run_m2_study(seeds=SEEDS, extractor=None, device: str | None = None,
                 verbose: bool = True) -> dict:
    """Evaluate M0, M1 and every M2 candidate across seeds; pick M2.

    Returns a payload with the full grids (per config), the seed-averaged
    blocks, and the aggregate M0/M1/M2 comparison.  Nothing here is written to
    disk -- :mod:`run_hw02` owns the file layout.
    """
    extractor = extractor or get_feature_extractor(device or get_device())
    device = device or get_device()

    m0_blocks, m1_blocks = [], []
    a_seeds, b_seeds, c_seeds = [], [], []
    for seed in seeds:
        if verbose:
            print(f"  [M2 study] seed {seed}: M0/M1 + variants A/B/C ...")
        m0_blocks.append(evaluate_m0(seed, extractor, device))
        m1_blocks.append(evaluate_m1(seed, extractor, device))
        a_seeds.append(evaluate_variant_a(seed, extractor, device))
        b_seeds.append(evaluate_variant_b(seed, extractor, device))
        c_seeds.append(evaluate_variant_c(seed, extractor, device))

    m0 = aggregate_blocks(m0_blocks)
    m1 = aggregate_blocks(m1_blocks)
    variant_a = aggregate_configs(a_seeds)
    variant_b = aggregate_configs(b_seeds)
    variant_c = aggregate_configs(c_seeds)

    a_best = best_config(variant_a)
    b_best = best_config(variant_b)
    c_best = best_config(variant_c)

    models = {
        "M0": m0,
        "M1": m1,
        f"M2_A({a_best})": variant_a[a_best],
        f"M2_B({b_best})": variant_b[b_best],
        f"M2_C({c_best})": variant_c[c_best],
    }
    comparison = compare_models(models)

    # M2 = the most stable candidate among those indistinguishable from the best,
    # where the candidates are the best config of each variant.
    candidates = {k: v for k, v in models.items() if k.startswith("M2_")}
    m2_key = best_config(candidates)
    m2 = {"key": m2_key, "block": candidates[m2_key],
          "variant_a_best": a_best, "variant_b_best": b_best,
          "variant_c_best": c_best}

    return {
        "geometry": {"strategy": GEOMETRY_STRATEGY, "interpolation": GEOMETRY_INTERP},
        "m1_recipe": {"spec": M1_SPEC_NAME, "replicates": M1_REPLICATES,
                      "targets": TARGET_LABELS},
        "M0": m0, "M1": m1,
        "variant_a": variant_a, "variant_b": variant_b, "variant_c": variant_c,
        "m2": m2,
        "models": models,
        "comparison": comparison,
    }
