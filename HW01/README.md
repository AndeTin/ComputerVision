# HW01 — few-shot input pipeline and custom augmentation

> [繁體中文版](README.zh-TW.md)

Produces **M0** (real images only) and **M1** (M0 + custom augmentation).
No generative model is used anywhere in this submission.

## Run

All commands are run from the **repository root** (`ComputerVision/`, one level
above this README). No absolute paths are required — every path in the code is
derived from the file's own location, so the repository can be cloned or moved
anywhere.

**macOS / Linux**

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python HW01/run_experiments.py                # Q1 + Q2 + Q3, all figures
python HW01/run_experiments.py --stage q1     # one stage
python HW01/run_experiments.py --no-plots     # skip figures
python HW01/run_experiments.py --no-samples   # skip the slow montage
```

**Windows (PowerShell)**

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python HW01\run_experiments.py
python HW01\run_experiments.py --stage q1
python HW01\run_experiments.py --no-plots
python HW01\run_experiments.py --no-samples
```

Windows notes:

- If PowerShell refuses to activate the venv, allow local scripts for the
  current session:
  `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`
- `.\.venv\Scripts\Activate.ps1` is the PowerShell path;
  `.venv\Scripts\activate.bat` is the `cmd.exe` one.
- The dataset is found relative to the repository, so it needs no configuration.
  If you keep the images elsewhere, set the location explicitly:
  `$env:HW01_DATASET_DIR = "D:\data\dataset"`
- No POSIX-only tooling is used, so results are identical on both platforms.

From-scratch run is ~23 min on CPU. Resumed runs are ~1 min: 512-d backbone
features are memoised under `HW01/.cache/`, keyed by everything that can change
them. Delete that folder to force recomputation.

## Environment

Pinned in `requirements.txt` (direct deps, repository root) and
`requirements-lock.txt` (exact venv). A CPU-only install is enough and skips
~4.5 GB of CUDA wheels:

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

The same command works in PowerShell. ResNet-18 ImageNet weights download on
first run (~45 MB) and cache to `~/.cache/torch` (`%USERPROFILE%\.cache\torch`
on Windows).

### Configuration

| Variable | Default | Purpose |
|----------|---------|---------|
| `HW01_DATASET_DIR` | `<repo root>/dataset` | location of the per-class folders |

## Layout

```
<repo root>/
  requirements.txt
  requirements-lock.txt
  dataset/               course data, one folder per class (cl01..cl20)
  HW01/
    hw01/                library — no stage logic
      data.py            splitting, EXIF-safe loading, geometry strategies, paths, dataset stats
      models.py          frozen ResNet-18 feature extractor + scikit-learn LogisticRegression probe
      metrics.py         per-run metrics, cross-seed mean ± sd, confusion matrices, interval-overlap tests
      augmentation.py    augmentation recipes, targeted replication, record expansion
      experiment.py      one code path from config → features → probe → metrics
      reporting.py       summary.json assembly and report-ready markdown tables
    stages/              one module per assignment question
      q1_input_pipeline.py         split rule, seed fixation order, EXIF audit, normalisation
      q2_resolution_comparison.py  3 strategies × 2 interpolations × 3 seeds, geometry audit
      q3_augmentation_design.py    M0 baseline, weak-class selection, integrity check, M1 sweep
    run_experiments.py   entry point: stage dispatch, summary, all figures
    results/
      summary.json       report-facing: headline numbers, rankings, markdown tables
      full/              verbose archive: per-seed detail, regenerates every figure
    plots/               7 figures
    .cache/              memoised features (gitignored)
```

The library carries no knowledge of which stage is running, and the stages carry
no figure code. `run_experiments.py` is the only module that knows both.

Every path is computed from `__file__`, so the project is location-independent
and behaves the same on Windows, macOS and Linux.

## Results layout

Two files, two audiences:

| Path | Contains | Use |
|------|----------|-----|
| `HW01/results/summary.json` | headline M0/M1 table, Q2 ranking, Q3 ratio sweep, per-class recall, pre-rendered markdown | quoting numbers in the write-up |
| `HW01/results/full/*.json` | every field including per-seed 20×20 confusion matrices | redrawing any figure without retraining |

`full/` is intentionally verbose — it is the reproducibility archive, not
something to read. A full run is ~3.5 min with a warm `.cache/`, ~23 min cold.

## Reproducibility contract

Seeds are fixed at `SEEDS = (42, 420, 4200)`, applied in a documented order:

1. `set_global_seed(seed)` — `random`, `numpy`, `torch`, `torch` (CUDA if present)
2. per-class split permutation: `random.Random(seed*1000 + class_index)`
3. DataLoader shuffling: separately seeded `torch.Generator`
4. augmentation RNG: `random.Random((seed*1000003 + index)*97 + replicate)`

Every reported number is the mean ± standard deviation over those three
configurations. Comparisons are phrased by whether the intervals overlap;
"significantly" is never used, because three seeds cannot support it.

`PYTHONHASHSEED` is **not** set from inside the process — CPython only reads it
at interpreter startup, so doing so would be a no-op. Hash-order determinism is
achieved structurally instead: every directory and filename listing is
explicitly `sorted()` before use, so no result depends on set or dict iteration
order. Verified by running under `PYTHONHASHSEED=0`, `1` and `12345` and
confirming byte-identical output. To pin it for external tooling, export it
before launching.

## Model

`torchvision.models.resnet18(weights=IMAGENET1K_V1)`, every parameter
`requires_grad=False`, `fc` replaced by `nn.Identity` to expose the 512-d
pooled feature. The classifier is `sklearn.linear_model.LogisticRegression`
(`solver='lbfgs'`, `C=1.0` held fixed across every model in the study). The
legacy `multi_class` argument was removed in scikit-learn 1.5 and is not used.

## Known caveats

- The dataset folder holds training images only, so the 3-per-class validation
  set is carved out of the provided files. Each class contributes 3 validation
  images, which is why per-class recall can only take the values
  {0, ⅓, ⅔, 1} and its standard deviation is coarse. This is the dominant
  source of seed-to-seed variance and is why the confidence intervals are wide.
- Only 32 of 292 images have a short side below 224 px, so the low-resolution
  subgroup in Q2 is small (n≈8 per seed). It is reported for direction only and
  is not used to rank strategies.
- `C=1.0` is not tuned. With 7–21 images per class the regularisation strength
  materially changes the decision boundary, making it a natural M2 lever.
- The geometry audit in Q2 is computed analytically from the transform
  definitions, not by scanning for dark pixels. An earlier pixel-scan version
  reported non-zero "padding" for `direct`, a strategy that injects none: it was
  counting genuinely black photograph content. The analytic version also
  separates the two failure modes the strategies trade off — fabricated zero
  fill versus discarded original content. `resize_crop` is the only strategy
  that does both.
- Verified on Linux with Python 3.12. The code is platform-independent and
  Windows instructions are given above, but the reported numbers have not been
  re-measured on Windows.
