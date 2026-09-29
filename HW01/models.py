"""Frozen ResNet-18 features + scikit-learn Logistic Regression probe.

The assignment fixes both halves of the model: the backbone must be
ImageNet-pretrained ResNet-18 with the convolutional stack frozen, and the
first submission's classifier must be ``sklearn.linear_model.LogisticRegression``.

Extracting 512-d features once and caching them makes the downstream probe
training effectively instantaneous, so the 3-seed x several-config sweep stays
tractable on CPU.
"""

from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights, resnet18

from data import CACHE_DIR, ensure_dirs


class ResNet18FeatureExtractor(nn.Module):
    """ImageNet ResNet-18 truncated after global average pooling.

    ``fc`` is replaced by ``nn.Identity`` so the module emits the 512-d
    pooled feature vector. Every parameter has ``requires_grad=False``: the
    backbone is a fixed function, not something the probe fine-tunes.
    """

    feature_dim = 512

    def __init__(self, weights=ResNet18_Weights.IMAGENET1K_V1):
        super().__init__()
        self.backbone = resnet18(weights=weights)
        self.backbone.fc = nn.Identity()
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def train(self, mode: bool = True):  # noqa: D102 - keep backbone in eval
        return super().train(False)


def get_feature_extractor(device: str = "cpu") -> ResNet18FeatureExtractor:
    ensure_dirs()
    model = ResNet18FeatureExtractor().to(device)
    model.eval()
    return model


@torch.no_grad()
def extract_features(
    model: ResNet18FeatureExtractor,
    loader: DataLoader,
    device: str = "cpu",
    cache_key: str | None = None,
    use_cache: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Run the backbone over a loader and return features, labels, groups, paths.

    When ``cache_key`` is given the arrays are memoised under ``.cache/``,
    keyed by everything that can change the features (seed, split, strategy,
    interpolation, augmentation spec).
    """
    path = os.path.join(CACHE_DIR, f"feat_{cache_key}.npz") if cache_key else None
    if use_cache and path and os.path.exists(path):
        z = np.load(path, allow_pickle=True)
        return z["X"], z["y"], z["g"], list(z["paths"])

    feats, labels, groups, paths = [], [], [], []
    for images, y, p, g in loader:
        feats.append(model(images.to(device)).cpu().numpy())
        labels.append(y.numpy())
        groups.append(np.asarray(g))
        paths.extend(p)

    X = np.vstack(feats).astype(np.float32)
    y = np.concatenate(labels)
    g = np.concatenate(groups)

    if use_cache and path:
        np.savez_compressed(path, X=X, y=y, g=g, paths=np.asarray(paths, dtype=object))
    return X, y, g, paths


class LogisticRegressionProbe:
    """Linear probe on frozen features.

    ``C`` is held fixed across every model in the study. With 7-21 training
    images per class the regularisation strength materially changes the
    decision boundary, so leaving it at scikit-learn's default is a choice
    that has to be stated rather than inherited silently. It is also the
    natural lever for the M2 submission.
    """

    def __init__(self, C: float = 1.0, seed: int = 0, max_iter: int = 5000):
        self.C = C
        # lbfgs is the solver that handles multinomial automatically; the
        # legacy `multi_class` argument was removed in scikit-learn 1.5+.
        self.clf = LogisticRegression(
            C=C, solver="lbfgs", max_iter=max_iter, random_state=seed
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegressionProbe":
        self.clf.fit(X, y)
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return self.clf.decision_function(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict(X)

    @property
    def n_iter(self) -> int:
        return int(np.max(self.clf.n_iter_))
