"""
A small 1D-ResNet ECG classifier for condition adherence ("does a classifier
trained on real ECGs recognise the requested labels in generated ones?") and
for TSTR / TRTR (train on synthetic / real, test on real).

Train it on data *separate* from the generator's training split, otherwise
adherence scores are optimistic.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ecgen.evaluation.features import ArrayLike
from ecgen.evaluation.measure import _as_numpy


class _Block(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(cin, cout, 7, stride, 3, bias=False)
        self.bn1 = nn.BatchNorm1d(cout)
        self.conv2 = nn.Conv1d(cout, cout, 7, 1, 3, bias=False)
        self.bn2 = nn.BatchNorm1d(cout)
        self.skip = (
            nn.Sequential(
                nn.Conv1d(cin, cout, 1, stride, bias=False), nn.BatchNorm1d(cout)
            )
            if (stride != 1 or cin != cout)
            else nn.Identity()
        )

    def forward(self, x: Tensor) -> Tensor:
        h = F.relu(self.bn1(self.conv1(x)))
        return F.relu(self.bn2(self.conv2(h)) + self.skip(x))


class ResNet1dClassifier(nn.Module):
    """``[B, C, T] -> [B, num_classes]`` logits; :meth:`features` gives the pooled embedding."""

    def __init__(
        self,
        in_channels: int = 12,
        num_classes: int = 5,
        width: int = 64,
        depth: int = 4,
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, width, 15, 2, 7, bias=False),
            nn.BatchNorm1d(width),
            nn.ReLU(),
            nn.MaxPool1d(3, 2, 1),
        )
        layers, ch = [], width
        for i in range(depth):
            out = width * min(2**i, 4)
            layers.append(_Block(ch, out, 1 if i == 0 else 2))
            ch = out
        self.layers = nn.Sequential(*layers)
        self.head = nn.Linear(ch, num_classes)
        self.embed_dim = ch

    def features(self, x: Tensor) -> Tensor:
        return self.layers(self.stem(x)).mean(-1)

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.features(x))


def fit_classifier(
    x: ArrayLike,
    y: ArrayLike,
    multilabel: bool = True,
    epochs: int = 20,
    lr: float = 1e-3,
    batch_size: int = 64,
    width: int = 64,
    device: Optional[str] = None,
    seed: int = 0,
) -> ResNet1dClassifier:
    """
    Train :class:`ResNet1dClassifier` on ``x [N, C, T]`` and labels ``y``
    (multi-hot ``[N, K]`` with ``multilabel=True``, else class indices ``[N]``).
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    xt = torch.as_tensor(_as_numpy(x), dtype=torch.float32)
    yt = torch.as_tensor(np.asarray(_as_numpy(y)))
    num_classes = yt.shape[1] if multilabel else int(yt.max()) + 1
    model = ResNet1dClassifier(xt.shape[1], num_classes, width).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, lr, total_steps=epochs * ((len(xt) + batch_size - 1) // batch_size)
    )
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(len(xt))
        for i in range(0, len(xt), batch_size):
            idx = perm[i : i + batch_size]
            logits = model(xt[idx].to(device))
            target = yt[idx].to(device)
            if multilabel:
                loss = F.binary_cross_entropy_with_logits(logits, target.float())
            else:
                loss = F.cross_entropy(logits, target.long())
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
    return model.eval()


@torch.no_grad()
def predict(
    model: nn.Module, x: ArrayLike, batch_size: int = 128, multilabel: bool = True
) -> np.ndarray:
    """Probabilities: sigmoid (multi-label) or softmax (single-label)."""
    device = next(model.parameters()).device
    xt = torch.as_tensor(_as_numpy(x), dtype=torch.float32)
    out = []
    model.eval()
    for i in range(0, len(xt), batch_size):
        logits = model(xt[i : i + batch_size].to(device)).float()
        out.append(
            (torch.sigmoid(logits) if multilabel else torch.softmax(logits, -1))
            .cpu()
            .numpy()
        )
    return np.concatenate(out)


def label_metrics(
    probs: np.ndarray, y: np.ndarray, multilabel: bool = True
) -> Dict[str, float]:
    """Macro AUROC over labels present with both values (multi-label), or accuracy + macro AUROC."""
    from sklearn.metrics import roc_auc_score

    y = np.asarray(y)
    if multilabel:
        aucs = [
            roc_auc_score(y[:, k], probs[:, k])
            for k in range(y.shape[1])
            if 0 < y[:, k].sum() < len(y)
        ]
        return {
            "macro_auroc": float(np.mean(aucs)) if aucs else float("nan"),
            "n_labels_scored": len(aucs),
        }
    acc = float((probs.argmax(1) == y).mean())
    try:
        auc = float(roc_auc_score(y, probs, multi_class="ovr"))
    except ValueError:
        auc = float("nan")
    return {"accuracy": acc, "macro_auroc": auc}


def label_adherence(
    model: nn.Module,
    generated: ArrayLike,
    requested: ArrayLike,
    multilabel: bool = True,
) -> Dict[str, float]:
    """Score generated ECGs against the labels they were generated for."""
    return label_metrics(
        predict(model, generated, multilabel=multilabel),
        _as_numpy(requested),
        multilabel,
    )


def tstr(
    train_x: ArrayLike,
    train_y: ArrayLike,
    test_x: ArrayLike,
    test_y: ArrayLike,
    multilabel: bool = True,
    **fit_kwargs,
) -> Dict[str, float]:
    """Train on ``train_*`` (synthetic for TSTR, real for TRTR), test on real ``test_*``."""
    model = fit_classifier(train_x, train_y, multilabel=multilabel, **fit_kwargs)
    return label_metrics(
        predict(model, test_x, multilabel=multilabel), _as_numpy(test_y), multilabel
    )
