from typing import cast

import torch.nn as nn
from torchvision.models import resnet50

NAME = "resnet"

def _adapt_first_conv(model, in_chans: int):
    if in_chans == 3:
        return model
    old_conv = model.conv1
    model.conv1 = nn.Conv2d(
        in_chans,
        old_conv.out_channels,
        kernel_size=cast(tuple[int, int], old_conv.kernel_size),
        stride=cast(tuple[int, int], old_conv.stride),
        padding=old_conv.padding,
        bias=old_conv.bias is not None,
    )
    return model


def create_model(input_shape, config):
    c, _, _ = input_shape
    net = resnet50(weights=None)
    net = _adapt_first_conv(net, c)
    num_ftrs = net.fc.in_features
    net.fc = nn.Linear(num_ftrs, config["num_classes"])
    return net
