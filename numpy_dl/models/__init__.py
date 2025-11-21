"""Pre-built model architectures."""

from numpy_dl.models.mlp import MLP
from numpy_dl.models.cnn import SimpleCNN, VGG
from numpy_dl.models.resnet import (
    ResNet, BasicBlock, Bottleneck,
    resnet18, resnet34, resnet50, resnet101, resnet152
)
from numpy_dl.models.unet import UNet
from numpy_dl.models.rnn_models import SimpleRNN, Seq2Seq
from numpy_dl.models.multitask import (
    TaskHead,
    HardParameterSharing,
    SoftParameterSharing,
    MultiTaskModel,
    create_hard_sharing_model,
    create_soft_sharing_model,
)

__all__ = [
    'MLP',
    'SimpleCNN',
    'VGG',
    'ResNet',
    'BasicBlock',
    'Bottleneck',
    'resnet18',
    'resnet34',
    'resnet50',
    'resnet101',
    'resnet152',
    'UNet',
    'SimpleRNN',
    'Seq2Seq',
    # Multi-task learning
    'TaskHead',
    'HardParameterSharing',
    'SoftParameterSharing',
    'MultiTaskModel',
    'create_hard_sharing_model',
    'create_soft_sharing_model',
]
