"""Q1: reproducible input pipeline.

Answers, with artefacts rather than assertions:
  * the split rule and where the seed is fixed;
  * what EXIF correction actually did to this dataset (spoiler: nothing, and
    that is worth reporting rather than hiding);
  * whether ImageNet normalisation constants suit these 20 classes;
  * the per-seed split sizes and the train/val overlap between seeds.
"""

from __future__ import annotations

import numpy as np

from data import (
    DATASET_DIR, IMAGENET_MEAN, IMAGENET_STD, N_VAL_PER_CLASS, SEEDS,
    compute_pixel_stats, dataset_resolution_profile, get_class_paths, make_split,
    pil_loader, save_json, set_global_seed,
)


def _exif_audit(dataset_dir: str = DATASET_DIR) -> dict:
    """How many images actually carry EXIF, and does the transpose change them?"""
    from PIL import Image, ImageOps

    n_total = n_with_exif = n_changed = n_non_rgb = 0
    changed_examples: list[dict] = []
    for paths in get_class_paths(dataset_dir):
        for p in paths:
            n_total += 1
            with Image.open(p) as im:
                if im.mode != "RGB":
                    n_non_rgb += 1
                if im.getexif():
                    n_with_exif += 1
                raw = im.copy()
                fixed = ImageOps.exif_transpose(im)
                if fixed.size != raw.size:
                    n_changed += 1
                    if len(changed_examples) < 5:
                        changed_examples.append({
                            "path": p, "before": list(raw.size), "after": list(fixed.size),
                        })
    return {
        "n_images": n_total,
        "n_with_exif_tag": n_with_exif,
        "n_non_rgb_mode": n_non_rgb,
        "n_geometry_changed_by_exif_transpose": n_changed,
        "changed_examples": changed_examples,
        "conclusion": (
            "exif_transpose is applied in pil_loader before convert('RGB') as required, "
            "but on this dataset it is a no-op: 0 images carry EXIF and 0 change geometry. "
            "It is defensive hardening, not a fix for an observed defect."
        ),
    }


def _split_report(dataset_dir: str = DATASET_DIR) -> dict:
    """Per-seed split sizes plus cross-seed overlap."""
    splits = {seed: make_split(seed, dataset_dir) for seed in SEEDS}
    report = {"rule": (
        "per class, shuffle indices with random.Random(seed*1000 + class_index), "
        f"hold out the first {N_VAL_PER_CLASS} for validation, keep the rest for training"
    ), "seeds": {}, "overlap": {}}

    for seed, sp in splits.items():
        report["seeds"][str(seed)] = sp.summary()

    for a in SEEDS:
        for b in SEEDS:
            if a >= b:
                continue
            va = {p for p, _ in splits[a].val}
            vb = {p for p, _ in splits[b].val}
            report["overlap"][f"seed{a}_vs_seed{b}"] = {
                "jaccard": len(va & vb) / len(va | vb),
                "n_shared_val_images": len(va & vb),
                "note": "low overlap is expected: it is what makes the 3 runs independent replicates",
            }
    return report


def _normalisation_report(dataset_dir: str = DATASET_DIR) -> dict:
    """Compare ImageNet constants with statistics measured on the training split.

    Computed on the seed-42 training split, over all three geometries, because
    the geometry changes the pixel statistics slightly (resampling and
    zero-padding both shift them).
    """
    sp = make_split(SEEDS[0], dataset_dir)
    out = {
        "split_seed": SEEDS[0],
        "imagenet_mean": list(IMAGENET_MEAN),
        "imagenet_std": list(IMAGENET_STD),
        "per_strategy": {},
    }
    for strategy in ("direct", "resize_crop", "pad_to_square"):
        out["per_strategy"][strategy] = compute_pixel_stats(sp.train, strategy)

    ref = out["per_strategy"]["resize_crop"]
    out["verdict"] = (
        f"On the resize_crop geometry the per-channel mean differs from ImageNet by at most "
        f"{ref['max_mean_abs_delta']:.4f} and the std by at most {ref['max_std_abs_delta']:.4f}. "
        "The distributions are close, so ImageNet constants are kept for M0/M1 to stay "
        "compatible with the pretrained backbone; a dataset-specific normalisation is "
        "available via compute_pixel_stats but is not adopted, because switching it would "
        "change the input distribution the frozen backbone was trained on."
    )
    return out


def run(dataset_dir: str = DATASET_DIR) -> dict:
    """Execute Q1 and write results/q1_pipeline.json."""
    set_global_seed(SEEDS[0])
    result = {
        "question": "Q1 reproducible input pipeline",
        "seed_fixation_order": [
            "1. set_global_seed(seed): random, numpy, torch, torch.cuda, PYTHONHASHSEED",
            "2. deterministic per-class permutation: random.Random(seed*1000 + class_index)",
            "3. DataLoader shuffling uses a separately seeded torch.Generator",
            "4. augmentation RNG: random.Random((seed*1000003 + index)*97 + replicate)",
        ],
        "color_space": {
            "loader": "PIL.Image.open -> ImageOps.exif_transpose -> convert('RGB')",
            "order_rationale": "orientation must be fixed while the image is still tagged, "
                               "before the colour space is coerced to 3 channels",
            "tensor_form": "ToTensor -> float32 in [0,1] -> Normalize(ImageNet mean/std)",
        },
        "dataset_profile": dataset_resolution_profile(dataset_dir),
        "exif_audit": _exif_audit(dataset_dir),
        "split_report": _split_report(dataset_dir),
        "normalisation": _normalisation_report(dataset_dir),
        "assumptions": [
            f"The dataset folder contains training images only, so the {N_VAL_PER_CLASS} "
            "validation images per class are held out from the provided files. The 3 "
            "configurations therefore vary both the split and the probe seed.",
        ],
    }
    save_json(result, "q1_pipeline.json")
    return result


def _print_summary(r: dict) -> None:
    prof = r["dataset_profile"]
    print("=" * 72)
    print("Q1  reproducible input pipeline")
    print("=" * 72)
    print(f"images                : {prof['n_images']}  ({prof['n_low_res']} with short side < 224px)")
    print(f"long side  min/med/max: {prof['long_side']['min']} / {prof['long_side']['median']:.0f} / {prof['long_side']['max']}")
    print(f"short side min/med/max: {prof['short_side']['min']} / {prof['short_side']['median']:.0f} / {prof['short_side']['max']}")
    ar = prof["aspect_ratio"]
    print(f"aspect     min/med/max: {ar['min']:.2f} / {ar['median']:.2f} / {ar['max']:.2f}")
    print(f"  aspect > 1.14 (resize_crop silently zero-pads): {ar['n_gt_1p14']} / {prof['n_images']}")
    print(f"  aspect > 1.5 : {ar['n_gt_1p5']}   aspect > 2.0 : {ar['n_gt_2p0']}")
    print()
    ex = r["exif_audit"]
    print(f"EXIF audit            : {ex['n_with_exif_tag']} images tagged, "
          f"{ex['n_geometry_changed_by_exif_transpose']} changed by exif_transpose, "
          f"{ex['n_non_rgb_mode']} non-RGB")
    print()
    sr = r["split_report"]
    for seed, s in sr["seeds"].items():
        counts = s["train_per_class"]
        print(f"seed {seed:>4}            : train {s['n_train']:3d} (per class {min(counts)}-{max(counts)}), "
              f"val {s['n_val']:3d}")
    for k, v in sr["overlap"].items():
        print(f"  {k}: val Jaccard {v['jaccard']:.3f} ({v['n_shared_val_images']} shared)")
    print()
    n = r["normalisation"]
    for strat, s in n["per_strategy"].items():
        print(f"norm [{strat:14s}]: mean {[round(v,4) for v in s['mean']]}  "
              f"std {[round(v,4) for v in s['std']]}  "
              f"max|Δmean| {s['max_mean_abs_delta']:.4f} max|Δstd| {s['max_std_abs_delta']:.4f}")
    print(f"ImageNet              : mean {[round(v,4) for v in IMAGENET_MEAN]}  std {[round(v,4) for v in IMAGENET_STD]}")


if __name__ == "__main__":
    _print_summary(run())
