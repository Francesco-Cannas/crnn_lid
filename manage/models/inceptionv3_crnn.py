from typing import cast

import torch
import torch.nn as nn
from torchvision.models import inception_v3

NAME = "Inceptionv3_CRNN"

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

class InceptionCRNN(nn.Module):
    def __init__(self, input_shape, num_classes: int) -> None:
        super().__init__()
        in_ch, _, _ = input_shape

        backbone = inception_v3(weights=None, aux_logits=False, transform_input=False)
        backbone = _adapt_first_conv(backbone, in_ch)
        for p in backbone.parameters():
            p.requires_grad = False
        self.backbone = backbone

        with torch.no_grad():
            x = torch.zeros(1, *input_shape)
            feat = self.backbone.features(x) if hasattr(self.backbone, "features") else self.backbone(x)
            _, c, h, w = feat.shape
            self.seq_len = w
            self.lstm_in = h * c

        self.lstm = nn.LSTM(
            input_size=self.lstm_in,
            hidden_size=512,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.classifier = nn.Linear(512 * 2, num_classes)

    def forward(self, x):
        x = self.backbone.features(x) if hasattr(self.backbone, "features") else self.backbone(x)
        # (B,C,H,W) → (B,W,H,C)
        x = x.permute(0, 3, 2, 1).contiguous().view(x.size(0), self.seq_len, self.lstm_in)
        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        return self.classifier(x)


def create_model(input_shape, config):
    return InceptionCRNN(input_shape, config["num_classes"])