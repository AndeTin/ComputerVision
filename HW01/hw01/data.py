"""Shared data utilities for HW01.

Handles: deterministic splitting, the three resize strategies (Q2), EXIF-safe
loading, and dataset statistics (Q1).

Design note on the validation split
-----------------------------------
The dataset ships 10-24 images per class with no separate validation folder,
so the 3-images-per-class validation set described in the assignment is carved
out of the given folders. This is the only way to honour the spec with the data
on disk, and it is recorded in results/ as an explicit assumption.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

# ----------------------------------------------------------------------------
# Global configuration
# ----------------------------------------------------------------------------

"""Root of the repository, i.e. the folder holding both ``HW01/`` and
``dataset/``. Derived from this file's location so the project works from any
checkout path, on any platform.
"""
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))   # HW01/hw01
HW01_DIR = os.path.dirname(PACKAGE_DIR)                     # HW01
RESULTS_DIR = os.path.join(HW01_DIR, "results")
FULL_RESULTS_DIR = os.path.join(RESULTS_DIR, "full")
SUMMARY_PATH = os.path.join(RESULTS_DIR, "summary.json")
PLOTS_DIR = os.path.join(HW01_DIR, "plots")
CACHE_DIR = os.path.join(HW01_DIR, ".cache")

#: The dataset lives beside ``HW01/`` in the repository root. Resolved relative
#: to this file rather than hardcoded, so moving or cloning the repo anywhere
#: works. Override with the ``HW01_DATASET_DIR`` environment variable if the
#: data is kept outside the checkout.
DATASET_DIR = os.environ.get(
    "HW01_DATASET_DIR", os.path.join(REPO_ROOT, "dataset"))

#: The three sampling configurations mandated by the assignment.
SEEDS = (42, 420, 4200)

NUM_CLASSES = 20
IMAGE_SIZE = 224
N_VAL_PER_CLASS = 3  # assignment states 3 validation images per class

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

#: Short side below this value means the image must be upscaled to reach the
#: 224px network input, so it carries no native detail at input scale.
LOW_RES_SHORT_SIDE = 224

RESIZE_BASE = 256  # longer-side target for the "resize + center crop" strategy
INTERPOLATIONS = ("bilinear", "bicubic")
STRATEGIES = ("direct", "resize_crop", "pad_to_square")


def ensure_dirs() -> None:
    for d in (RESULTS_DIR, FULL_RESULTS_DIR, PLOTS_DIR, CACHE_DIR):
        os.makedirs(d, exist_ok=True)


def set_global_seed(seed: int) -> None:
    """Seed every RNG the pipeline can touch, in a fixed documented order.

    Note on ``PYTHONHASHSEED``: setting it here is a no-op, because CPython
    reads it once at interpreter startup. Hash-order determinism is instead
    achieved structurally -- every directory and filename listing is explicitly
    ``sorted()`` before use, so no result depends on set or dict iteration
    order. If hash randomisation must be pinned for some external tool, export
    it before launching: ``PYTHONHASHSEED=0 python HW01/run_experiments.py``.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ----------------------------------------------------------------------------
# Path collection
# ----------------------------------------------------------------------------


def get_class_paths(dataset_dir: str = DATASET_DIR) -> list[list[str]]:
    """Return sorted per-class lists of image paths, ordered cl01..cl20.

    Sorting both the class names and the file names is part of the
    reproducibility contract: it removes filesystem-order dependence.
    """
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(
            f"dataset directory not found: {dataset_dir}\n"
            f"Expected one subfolder per class (cl01..cl{NUM_CLASSES:02d}) beside "
            f"HW01/, i.e. at {os.path.join(REPO_ROOT, 'dataset')}.\n"
            f"Set HW01_DATASET_DIR to point at it if the data lives elsewhere."
        )
    class_dirs = sorted(
        d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))
    )
    out = []
    for d in class_dirs:
        files = sorted(
            f
            for f in os.listdir(os.path.join(dataset_dir, d))
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        )
        out.append([os.path.join(dataset_dir, d, f) for f in files])
    return out


def read_size(path: str) -> tuple[int, int]:
    """Return (width, height) without decoding the pixel data."""
    with Image.open(path) as im:
        return im.size


def pil_loader(path: str) -> Image.Image:
    """Load a JPEG as RGB with EXIF orientation applied first.

    Q1 requirement: ``ImageOps.exif_transpose`` must run *before*
    ``convert("RGB")`` so that rotated camera images are upright and so that a
    palette/CMYK source is interpreted in the right colour space.
    """
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)  # 1. fix orientation
        return im.convert("RGB")  # 2. then normalise colour space


# ----------------------------------------------------------------------------
# Splitting
# ----------------------------------------------------------------------------


@dataclass
class Split:
    """A deterministic train/val split for one seed."""

    seed: int
    class_names: list[str]
    train: list[tuple[str, int]] = field(default_factory=list)
    val: list[tuple[str, int]] = field(default_factory=list)

    def summary(self) -> dict:
        train_per_class = [0] * len(self.class_names)
        for _, y in self.train:
            train_per_class[y] += 1
        return {
            "seed": self.seed,
            "n_train": len(self.train),
            "n_val": len(self.val),
            "train_per_class": train_per_class,
        }


def make_split(seed: int, dataset_dir: str = DATASET_DIR,
               n_val_per_class: int = N_VAL_PER_CLASS) -> Split:
    """Hold out exactly ``n_val_per_class`` images per class for validation.

    The per-class permutation uses ``seed + class_index`` so that classes are
    shuffled independently yet the whole split remains a pure function of
    ``seed``. ``N_VAL_PER_CLASS`` is subtracted directly (not computed as a
    ratio) so that every class yields a validation set of the same size, which
    is what the assignment specifies.
    """
    set_global_seed(seed)
    per_class = get_class_paths(dataset_dir)
    class_names = [os.path.basename(os.path.dirname(p[0])) for p in per_class]

    split = Split(seed=seed, class_names=class_names)
    for y, paths in enumerate(per_class):
        idx = list(range(len(paths)))
        random.Random(seed * 1000 + y).shuffle(idx)
        n_val = min(n_val_per_class, max(1, len(paths) - 1))
        val_idx, train_idx = idx[:n_val], idx[n_val:]
        split.val += [(paths[i], y) for i in val_idx]
        split.train += [(paths[i], y) for i in train_idx]
    return split


# ----------------------------------------------------------------------------
# Resolution grouping (Q2)
# ----------------------------------------------------------------------------


def resolution_group(path: str) -> str:
    """Group images by whether their short side reaches the network input size.

    ``low``  -> short side < 224px, so the image is upscaled and contributes
                interpolated (non-native) detail at input resolution.
    ``high`` -> short side >= 224px.
    """
    w, h = read_size(path)
    return "low" if min(w, h) < LOW_RES_SHORT_SIDE else "high"


def dataset_resolution_profile(dataset_dir: str = DATASET_DIR) -> dict:
    """Histogram of long side, short side and aspect ratio over the whole set."""
    longs, shorts, ars = [], [], []
    for paths in get_class_paths(dataset_dir):
        for p in paths:
            w, h = read_size(p)
            longs.append(max(w, h))
            shorts.append(min(w, h))
            ars.append(max(w, h) / min(w, h))
    longs, shorts, ars = map(np.asarray, (longs, shorts, ars))
    return {
        "n_images": int(longs.size),
        "long_side": {
            "min": int(longs.min()), "median": float(np.median(longs)), "max": int(longs.max())
        },
        "short_side": {
            "min": int(shorts.min()), "median": float(np.median(shorts)), "max": int(shorts.max())
        },
        "aspect_ratio": {
            "min": float(ars.min()), "median": float(np.median(ars)), "max": float(ars.max()),
            "n_gt_1p14": int((ars > RESIZE_BASE / IMAGE_SIZE).sum()),
            "n_gt_1p5": int((ars > 1.5).sum()),
            "n_gt_2p0": int((ars > 2.0).sum()),
        },
        "n_low_res": int((shorts < LOW_RES_SHORT_SIDE).sum()),
    }


# ----------------------------------------------------------------------------
# Transforms (Q2)
# ----------------------------------------------------------------------------

_PIL_FILTERS = {
    "bilinear": Image.BILINEAR,
    "bicubic": Image.BICUBIC,
}


class PadToSquare:
    """Aspect-preserving scale-to-fit, then zero-pad the short dimension.

    The long side is mapped to ``size``; the remaining short-side gap is split
    evenly between the two borders and filled with 0. Unlike
    ``Resize(256)+CenterCrop(224)`` this never lets torchvision decide the
    padding implicitly, so the amount of injected black is explicit and
    measurable.
    """

    def __init__(self, size: int = IMAGE_SIZE, interpolation: str = "bilinear"):
        self.size = size
        self.filter = _PIL_FILTERS[interpolation]

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        scale = self.size / max(w, h)
        new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
        img = img.resize((new_w, new_h), self.filter)
        pad_w, pad_h = self.size - new_w, self.size - new_h
        left, top = pad_w // 2, pad_h // 2
        padded = Image.new("RGB", (self.size, self.size), (0, 0, 0))
        padded.paste(img, (left, top))
        return padded

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"PadToSquare(size={self.size}, filter={self.filter})"


def build_geometry(strategy: str, interpolation: str = "bilinear"):
    """Return the geometry stage for one of the three Q2 strategies.

    - ``direct``        : anisotropic squash to 224x224, no synthetic pixels.
    - ``resize_crop``   : scale long side to 256 then centre-crop 224. Note that
                          torchvision's ``CenterCrop`` zero-pads whenever the
                          scaled short side is < 224, i.e. aspect ratio > 1.14.
    - ``pad_to_square`` : scale long side to 224 preserving ratio, zero-pad.
    """
    mode = InterpolationMode.BILINEAR if interpolation == "bilinear" else InterpolationMode.BICUBIC

    if strategy == "direct":
        return transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=mode)
    if strategy == "resize_crop":
        return transforms.Compose(
            [transforms.Resize(RESIZE_BASE, interpolation=mode), transforms.CenterCrop(IMAGE_SIZE)]
        )
    if strategy == "pad_to_square":
        return PadToSquare(IMAGE_SIZE, interpolation)
    raise ValueError(f"unknown strategy {strategy!r}")


def build_transform(
    strategy: str = "resize_crop",
    interpolation: str = "bilinear",
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
    train_augment: callable | None = None,
) -> transforms.Compose:
    """Full transform: geometry -> optional augmentation -> tensor -> normalise.

    Augmentation is applied in PIL space *before* the geometric resize so that
    rotated/photometric jitter is not re-interpolated twice.
    """
    stages: list = []
    if train_augment is not None:
        stages.append(transforms.Lambda(train_augment))
    stages += [
        build_geometry(strategy, interpolation),
        transforms.ToTensor(),
        transforms.Normalize(mean=list(mean), std=list(std)),
    ]
    return transforms.Compose(stages)


# ----------------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------------


class FewShotDataset(Dataset):
    """Yields ``(tensor, label, path, group)`` so metrics can be sliced later."""

    def __init__(self, records: list[tuple[str, int]], transform):
        self.records = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int):
        path, label = self.records[i]
        img = pil_loader(path)
        return self.transform(img), label, path, resolution_group(path)


# ----------------------------------------------------------------------------
# Dataset statistics (Q1)
# ----------------------------------------------------------------------------


def compute_pixel_stats(records: list[tuple[str, int]], strategy: str,
                        interpolation: str = "bilinear") -> dict:
    """Mean/std of the training pixels in [0,1] space, *before* normalisation.

    This is the quantity that must be compared against the ImageNet constants,
    since comparing post-normalisation tensors would be meaningless.
    """
    geom = build_geometry(strategy, interpolation)
    acc = np.zeros(3, dtype=np.float64)
    acc_sq = np.zeros(3, dtype=np.float64)
    n = 0
    for path, _ in records:
        arr = np.asarray(geom(pil_loader(path)), dtype=np.float64) / 255.0  # HxWx3
        acc += arr.reshape(-1, 3).sum(0)
        acc_sq += (arr.reshape(-1, 3) ** 2).sum(0)
        n += arr.shape[0] * arr.shape[1]

    mean = acc / n
    var = np.maximum(acc_sq / n - mean**2, 0.0)
    std = np.sqrt(var)
    return {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "n_pixels": int(n),
        "n_images": len(records),
        "imagenet_mean": list(IMAGENET_MEAN),
        "imagenet_std": list(IMAGENET_STD),
        "mean_abs_delta": np.abs(mean - np.asarray(IMAGENET_MEAN)).tolist(),
        "std_abs_delta": np.abs(std - np.asarray(IMAGENET_STD)).tolist(),
        "max_mean_abs_delta": float(np.abs(mean - np.asarray(IMAGENET_MEAN)).max()),
        "max_std_abs_delta": float(np.abs(std - np.asarray(IMAGENET_STD)).max()),
    }


# ----------------------------------------------------------------------------
# Loaders
# ----------------------------------------------------------------------------


def make_loader(records, transform, batch_size: int = 32, shuffle: bool = False,
                num_workers: int = 0) -> DataLoader:
    """Deterministic loader: the generator is seeded so shuffling repeats."""
    generator = torch.Generator()
    generator.manual_seed(0)
    return DataLoader(
        FewShotDataset(records, transform),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=generator if shuffle else None,
        drop_last=False,
    )


# ----------------------------------------------------------------------------
# Cache helpers
# ----------------------------------------------------------------------------


def cache_key(**parts) -> str:
    """Stable short hash over the arguments that determine a feature matrix."""
    blob = json.dumps(parts, sort_keys=True, default=str).encode()
    return hashlib.sha1(blob).hexdigest()[:16]


def save_json(obj, filename: str, full: bool = True) -> str:
    """Write a JSON artefact, by default to the verbose ``results/full/``.

    ``full=True`` keeps every field, including the per-seed confusion matrices
    needed to redraw the matrices, so any figure can be rebuilt without
    retraining. The report-facing view is assembled separately by
    :func:`hw01.reporting.build_summary` into ``results/summary.json``.
    """
    ensure_dirs()
    path = os.path.join(FULL_RESULTS_DIR if full else RESULTS_DIR, filename)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, default=str)
    return path
