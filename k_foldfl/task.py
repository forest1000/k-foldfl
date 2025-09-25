from collections import OrderedDict, defaultdict
from typing import Dict, Any, List, Sequence
import yaml
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models


def make_model(name: str) -> nn.Module:
    if name.lower() == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, 10)
        return model

    raise ValueError(f"Unsupported model: {name}")

_NORM_MEAN = (0.491, 0.482, 0.447)
_NORM_STD  = (0.247, 0.243, 0.262)

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(_NORM_MEAN, _NORM_STD)
])


# --- テストデータ用のトランスフォーム（データ拡張なし）---
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(_NORM_MEAN, _NORM_STD)
])

def load_cifar10(root: str):
    return datasets.CIFAR10(root, train=True, download=True, transform=train_transform)

def load_cifar10_test(root: str):
    return datasets.CIFAR10(root, train=False, download=True, transform=test_transform)

# ─────────── Dirichlet Non-IID split ───────────

def dirichlet_partition(
    train_idx: Sequence[int],
    labels: Sequence[int],
    num_clients: int,
    beta: float,
    seed: int = 42,
) -> List[List[int]]:
    """
    CIFAR-10 の train_idx 上で Dirichlet(β) による Non-IID 分割を行い、
    num_clients 個のインデックス集合を返す。
    """
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    train_idx = np.asarray(train_idx)

    num_classes = int(labels.max()) + 1
    idx_by_class = [train_idx[labels[train_idx] == y] for y in range(num_classes)]
    parts = [[] for _ in range(num_clients)]

    for c in range(num_classes):
        idx_c = idx_by_class[c]
        rng.shuffle(idx_c)
        # クライアントごとの比率を Dirichlet からサンプル
        props = rng.dirichlet(alpha=[beta] * num_clients)
        # 比率に応じて分割
        cuts = (np.cumsum(props) * len(idx_c)).astype(int)[:-1]
        split = np.split(idx_c, cuts)
        for i, chunk in enumerate(split):
            parts[i].extend(chunk.tolist())

    # シャッフルして返す
    for p in parts:
        rng.shuffle(p)
    return parts

# ─────────── Torch<->NumPy weights ───────────

def get_weights(net: nn.Module):
    return [val.detach().cpu().numpy() for _, val in net.state_dict().items()]

def set_weights(net: nn.Module, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

# ─────────── YAML helpers ───────────

def load_yaml(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)
