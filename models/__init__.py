from .common import make_conv, make_linear
from .lenet5 import LeNet5
from .resnet_cifar import ResNetCifar, resnet20, resnet32

MODELS = {"lenet5": LeNet5, "resnet20": resnet20, "resnet32": resnet32}


def build_model(name: str, **kwargs):
    if name not in MODELS:
        raise ValueError(f"unknown model: {name}")
    return MODELS[name](**kwargs)
