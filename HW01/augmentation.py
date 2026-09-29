"""Custom few-shot augmentation for M1 (Q3).

Two things distinguish this from a stock ``transforms.AutoAugment``:

1. **Targeted.** Augmentation is applied *only* to the classes M0 gets wrong.
   With 7-21 images per class, spreading extra samples over all 20 classes
   spends the budget where it is not needed; concentrating it on the weak tail
   is what the assignment asks for.
2. **Checked.** Every augmented sample is scored for semantic integrity by
   comparing its backbone feature against its own source image, so "did this
   augmentation destroy the image?" gets a number instead of an opinion.

All ops run in PIL space on uint8 images, before the geometric resize, and are
driven by an explicitly seeded RNG so that (image, replicate) is reproducible.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, asdict

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from torch.utils.data import Dataset

from data import resolution_group

#: A record is (path, label, replicate_index). replicate_index == 0 is the
#: untouched original, so a targeted class has total size 1 + n_replicates.
Record3 = tuple[str, int, int]


@dataclass(frozen=True)
class AugmentSpec:
    """A named, reproducible augmentation recipe.

    ``strength`` scales the geometric jitter. ``cutout_frac`` is the side
    fraction of the erased square; 0 disables cutout. ``sharpen`` counters the
    loss of high-frequency detail caused by upsampling small images.
    """

    name: str
    rotate_deg: float = 10.0
    translate_frac: float = 0.04
    scale_jitter: float = 0.10
    brightness: float = 0.15
    contrast: float = 0.15
    saturation: float = 0.15
    cutout_frac: float = 0.0
    sharpen: bool = False
    hflip: bool = True
    strength: float = 1.0

    def to_dict(self) -> dict:
        return asdict(self)


SPECS: dict[str, AugmentSpec] = {
    "none": AugmentSpec("none", rotate_deg=0.0, translate_frac=0.0, scale_jitter=0.0,
                        brightness=0.0, contrast=0.0, saturation=0.0, hflip=False),
    "mild": AugmentSpec("mild"),
    "mild_cutout": AugmentSpec("mild_cutout", cutout_frac=0.25),
    "mild_sharpen": AugmentSpec("mild_sharpen", sharpen=True),
    "strong": AugmentSpec("strong", rotate_deg=20.0, translate_frac=0.08, scale_jitter=0.20,
                          brightness=0.25, contrast=0.25, saturation=0.25,
                          cutout_frac=0.25, sharpen=True),
}


# ----------------------------------------------------------------------------
# Primitive ops (PIL, uint8)
# ----------------------------------------------------------------------------


def _rotate(img: Image.Image, deg: float) -> Image.Image:
    return img.rotate(deg, resample=Image.BILINEAR, expand=False, fillcolor=(0, 0, 0))


def _translate(img: Image.Image, frac: float, rng: random.Random) -> Image.Image:
    w, h = img.size
    dx = rng.randint(-int(frac * w), int(frac * w))
    dy = rng.randint(-int(frac * h), int(frac * h))
    out = Image.new("RGB", (w, h), (0, 0, 0))
    out.paste(img, (dx, dy))
    return out


def _scale(img: Image.Image, factor: float) -> Image.Image:
    w, h = img.size
    nw, nh = max(1, round(w * factor)), max(1, round(h * factor))
    small = img.resize((nw, nh), Image.BILINEAR)
    out = Image.new("RGB", (w, h), (0, 0, 0))
    out.paste(small, ((w - nw) // 2, (h - nh) // 2))
    return out


def _cutout(img: Image.Image, frac: float, rng: random.Random) -> Image.Image:
    """Erase one square, filled with the image mean colour.

    Mean-fill rather than black, so the probe is not handed "black square" as a
    spurious class cue -- relevant here because two of the three geometry
    strategies already inject black borders.
    """
    w, h = img.size
    side = max(1, int(frac * min(w, h)))
    x0 = rng.randint(0, max(0, w - side))
    y0 = rng.randint(0, max(0, h - side))
    fill = tuple(np.asarray(img, np.float32).reshape(-1, 3).mean(0).astype(np.uint8).tolist())
    ImageDraw.Draw(img).rectangle([x0, y0, x0 + side, y0 + side], fill=fill)
    return img


def apply_augmentation(img: Image.Image, spec: AugmentSpec, rng: random.Random) -> Image.Image:
    """Apply ``spec`` to a PIL image using the supplied RNG."""
    if spec.name == "none" or spec.strength == 0:
        return img

    s = spec.strength
    if spec.hflip and rng.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    if spec.rotate_deg > 0:
        img = _rotate(img, rng.uniform(-spec.rotate_deg, spec.rotate_deg) * s)
    if spec.translate_frac > 0:
        img = _translate(img, spec.translate_frac * s, rng)
    if spec.scale_jitter > 0:
        img = _scale(img, 1.0 + rng.uniform(-spec.scale_jitter, spec.scale_jitter) * s)
    if spec.brightness > 0:
        f = 1 + rng.uniform(-spec.brightness, spec.brightness) * s
        img = Image.fromarray(np.clip(np.asarray(img, np.float32) * f, 0, 255).astype(np.uint8))
    if spec.contrast > 0:
        arr = np.asarray(img, np.float32)
        f = 1 + rng.uniform(-spec.contrast, spec.contrast) * s
        img = Image.fromarray(np.clip((arr - arr.mean()) * f + arr.mean(), 0, 255).astype(np.uint8))
    if spec.saturation > 0:
        arr = np.asarray(img, np.float32)
        grey = arr.mean(axis=2, keepdims=True)
        f = 1 + rng.uniform(-spec.saturation, spec.saturation) * s
        img = Image.fromarray(np.clip(grey + (arr - grey) * f, 0, 255).astype(np.uint8))
    if spec.cutout_frac > 0:
        img = _cutout(img, spec.cutout_frac, rng)
    if spec.sharpen:
        img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=60, threshold=3))
    return img


# ----------------------------------------------------------------------------
# Record expansion + dataset
# ----------------------------------------------------------------------------


def expand_records(records: list[tuple[str, int]], target_labels, n_replicates: int) -> list[Record3]:
    """Replicate target-class images ``n_replicates`` times each.

    Replicate 0 is the pristine original, so the per-class augmentation ratio
    for a targeted class is exactly ``1 + n_replicates`` and untouched classes
    stay at 1x. This keeps the ratio interpretable when only part of the label
    space is augmented.
    """
    targets = set(target_labels or ())
    out: list[Record3] = []
    for path, y in records:
        out.append((path, y, 0))
        if y in targets:
            for r in range(1, n_replicates + 1):
                out.append((path, y, r))
    return out


def per_class_totals(records3: list[Record3], num_classes: int = 20) -> list[int]:
    totals = [0] * num_classes
    for _, y, _ in records3:
        totals[y] += 1
    return totals


class AugmentedFewShotDataset(Dataset):
    """Yields the same ``(tensor, label, path, group)`` tuple as ``FewShotDataset``.

    ``replicate_index > 0`` triggers augmentation for that sample. The RNG is
    seeded from ``(seed, index, replicate_index)`` so the same virtual sample is
    identical across reruns while different replicates still differ.
    """

    def __init__(self, records3: list[Record3], transform, spec: AugmentSpec,
                 target_labels, seed: int):
        self.records3 = records3
        self.transform = transform
        self.spec = spec
        self.target_labels = set(target_labels or ())
        self.seed = seed

    def __len__(self) -> int:
        return len(self.records3)

    def __getitem__(self, i: int):
        path, label, rep = self.records3[i]
        from data import pil_loader  # local import keeps data.py import-cycle free

        img = pil_loader(path)
        if rep > 0 and self.spec.name != "none" and label in self.target_labels:
            rng = random.Random((self.seed * 1_000_003 + i) * 97 + rep)
            img = apply_augmentation(img, self.spec, rng)
        return self.transform(img), label, path, resolution_group(path)

    def summary(self, class_names: list[str], num_classes: int = 20) -> dict:
        totals = per_class_totals(self.records3, num_classes)
        base = [0] * num_classes
        for _, y, rep in self.records3:
            if rep == 0:
                base[y] += 1
        return {
            "spec": self.spec.to_dict(),
            "target_class_names": [class_names[i] for i in sorted(self.target_labels)],
            "n_total": len(self.records3),
            "n_augmented": sum(1 for _, _, r in self.records3 if r > 0),
            "per_class_base": base,
            "per_class_total": totals,
            "per_class_ratio": [round(t / b, 2) if b else None for t, b in zip(totals, base)],
        }
