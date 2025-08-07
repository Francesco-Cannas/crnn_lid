from typing import cast

import torch.nn as nn
from torchvision.models import inception_v3, Inception_V3_Weights

NAME = "InceptionV3"

def _adapt_first_conv(net: nn.Module, in_chans: int) -> nn.Module:
    if in_chans == 3:
        return net

    old = net.Conv2d_1a_3x3.conv

    net.Conv2d_1a_3x3.conv = nn.Conv2d(
        in_chans,
        old.out_channels,
        kernel_size=cast(tuple[int, int], old.kernel_size),
        stride=cast(tuple[int, int], old.stride),
        padding=old.padding,
        bias=old.bias is not None,
    )
    return net


def create_model(input_shape, config):
    in_ch, _, _ = input_shape

    transform_input = (in_ch == 3)
    net = inception_v3(
        weights=Inception_V3_Weights.DEFAULT,
        aux_logits=True,
        transform_input=transform_input
    )

    net = _adapt_first_conv(net, in_ch)

    net.AuxLogits = None
    net.aux_logits = False

    num_ftrs = net.fc.in_features
    net.fc = nn.Linear(num_ftrs, config["num_classes"])

    return net
