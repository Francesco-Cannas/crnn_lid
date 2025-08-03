import torch.nn as nn
from torchvision.models import inception_v3

NAME = "InceptionV3"

def _adapt_first_conv(net: nn.Module, in_chans: int) -> nn.Module:
    if in_chans == 3:
        return net
    old = net.Conv2d_1a_3x3.conv
    net.Conv2d_1a_3x3.conv = nn.Conv2d(
        in_chans,
        old.out_channels,
        kernel_size=old.kernel_size,
        stride=old.stride,
        padding=old.padding,
        bias=old.bias is not None,
    )
    return net


def create_model(input_shape, config):
    in_ch, _, _ = input_shape
    net = inception_v3(weights=None, aux_logits=False, transform_input=False)
    net = _adapt_first_conv(net, in_ch)


    num_ftrs = net.fc.in_features
    net.fc = nn.Linear(num_ftrs, config["num_classes"])
    return net