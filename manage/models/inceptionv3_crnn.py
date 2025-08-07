from typing import Tuple, cast

import torch
import torch.nn as nn
from torchvision.models import inception_v3, Inception_V3_Weights

NAME = "Inceptionv3_CRNN"


def _adapt_first_conv(net, in_chans) -> nn.Module:
    if in_chans == 3:
        return net
    old_conv = net.Conv2d_1a_3x3.conv
    net.Conv2d_1a_3x3.conv = nn.Conv2d(
        in_chans,
        old_conv.out_channels,
        kernel_size=cast(tuple[int, int], old_conv.kernel_size),
        stride=cast(tuple[int, int], old_conv.stride),
        padding=old_conv.padding,
        bias=old_conv.bias is not None,
    )
    return net

class InceptionCRNN(nn.Module):
    def __init__(self, input_shape: Tuple[int, int, int], num_classes: int) -> None:
        super().__init__()
        in_ch, _, _ = input_shape

        backbone = inception_v3(
            weights=Inception_V3_Weights.DEFAULT,
            aux_logits=True,
            transform_input=False
        )

        backbone = _adapt_first_conv(backbone, in_ch)
        for param in backbone.parameters():
            param.requires_grad = False

        self.features = nn.Sequential(
            backbone.Conv2d_1a_3x3,
            backbone.Conv2d_2a_3x3,
            backbone.Conv2d_2b_3x3,
            backbone.maxpool1,
            backbone.Conv2d_3b_1x1,
            backbone.Conv2d_4a_3x3,
            backbone.maxpool2,
            backbone.Mixed_5b,
            backbone.Mixed_5c,
            backbone.Mixed_5d,
            backbone.Mixed_6a,
            backbone.Mixed_6b,
            backbone.Mixed_6c,
            backbone.Mixed_6d,
            backbone.Mixed_6e,
            backbone.Mixed_7a,
            backbone.Mixed_7b,
            backbone.Mixed_7c,
        )

        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            feat = self.features(dummy)
            _, c, h, w = feat.shape
            self.seq_len = w
            self.lstm_in = c * h

        self.lstm = nn.LSTM(
            input_size=self.lstm_in,
            hidden_size=512,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.classifier = nn.Linear(512 * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        seq = feat.permute(0, 3, 2, 1).contiguous().view(
            x.size(0), self.seq_len, self.lstm_in
        )
        self.lstm.flatten_parameters()
        out, _ = self.lstm(seq)
        return self.classifier(out[:, -1, :])

def create_model(input_shape, config):
    return InceptionCRNN(input_shape, config["num_classes"])
