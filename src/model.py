"""Model definition: a pretrained CNN (ResNet) with a new final layer for our brands.

Transfer learning idea:
    A ResNet trained on ImageNet (1.2 million images) has already learned general visual
    features: edges, curves, textures, wheels, headlights, grilles ...
    We keep those features and only replace the LAST layer (1000 ImageNet classes) with a
    new layer that has one output per car brand (20). Then we train on our own images.
"""

import torch.nn as nn
from torchvision import models

SUPPORTED_ARCHS = ("resnet18", "resnet50")


def build_model(arch: str, n_classes: int, pretrained: bool = True) -> nn.Module:
    """Create a ResNet whose final layer outputs n_classes values.

    pretrained=True  loads the ImageNet weights (used for training).
    pretrained=False creates the same network with random weights (used by the API,
                     which then loads our own trained weights from a file).
    """
    if arch == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
    elif arch == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
    else:
        raise ValueError(f"Unsupported architecture '{arch}'. Choose from {SUPPORTED_ARCHS}.")

    # Replace the last fully connected layer: (features) -> n_classes scores.
    model.fc = nn.Linear(model.fc.in_features, n_classes)
    return model


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    """Freeze (False) or unfreeze (True) every layer except the new final layer 'fc'.

    A frozen layer keeps its weights during training.
    """
    for name, param in model.named_parameters():
        if not name.startswith("fc."):
            param.requires_grad = trainable