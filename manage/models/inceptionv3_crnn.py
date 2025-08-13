from typing import Tuple, List, cast

import torch
import torch.nn as nn
from torch import Tensor
from torchvision.models import inception_v3, Inception_V3_Weights

NAME = "Inceptionv3_CRNN"


def _init_and_adapt_first_conv(net: nn.Module, in_chans: int) -> nn.Module:
    if in_chans == 3:
        return net
    old = net.Conv2d_1a_3x3.conv
    new = nn.Conv2d(
        in_chans,
        old.out_channels,
        kernel_size=cast(tuple[int, int], old.kernel_size),
        stride=cast(tuple[int, int], old.stride),
        padding=old.padding,
        bias=old.bias is not None,
    )
    with torch.no_grad():
        base = cast(Tensor, old.weight.mean(dim=1, keepdim=True))
        scaled = base.repeat(1, in_chans, 1, 1) * (3.0 / float(in_chans))
        new.weight.copy_(scaled)
        if old.bias is not None:
            new.bias.copy_(old.bias)
    net.Conv2d_1a_3x3.conv = new
    return net


def _set_requires_grad(module: nn.Module, requires_grad: bool) -> None:
    for p in module.parameters():
        p.requires_grad = requires_grad


def _collect_backbone_blocks(backbone) -> List[nn.Module]:
    return [
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
    ]


def _block_names_sequence() -> List[str]:
    return [
        "Conv2d_1a_3x3",
        "Conv2d_2a_3x3",
        "Conv2d_2b_3x3",
        "maxpool1",
        "Conv2d_3b_1x1",
        "Conv2d_4a_3x3",
        "maxpool2",
        "Mixed_5b",
        "Mixed_5c",
        "Mixed_5d",
        "Mixed_6a",
        "Mixed_6b",
        "Mixed_6c",
        "Mixed_6d",
        "Mixed_6e",
        "Mixed_7a",
        "Mixed_7b",
        "Mixed_7c",
    ]


class InceptionCRNN(nn.Module):
    def __init__(
            self,
            input_shape: Tuple[int, int, int],
            num_classes: int,
            *,
            transform_input: bool = False,
            freeze_backbone: bool = True,
            unfreeze_from: str = "Mixed_6e",
            pool_h: int = 4,
            lstm_hidden: int = 384,
            lstm_layers: int = 1,
            bidirectional: bool = True,
            weights: Inception_V3_Weights | None = Inception_V3_Weights.DEFAULT,
    ) -> None:
        super().__init__()
        in_ch, _, _ = input_shape
        backbone = inception_v3(
            weights=weights,
            aux_logits=True,
            transform_input=transform_input,
        )
        backbone = _init_and_adapt_first_conv(backbone, in_ch)
        blocks = _collect_backbone_blocks(backbone)
        names = _block_names_sequence()
        if freeze_backbone:
            _set_requires_grad(backbone, False)
            backbone.eval()
        if unfreeze_from not in names:
            unfreeze_idx = names.index("Mixed_6e")
        else:
            unfreeze_idx = names.index(unfreeze_from)
        self._frozen_modules: List[nn.Module] = []
        self._trainable_modules: List[nn.Module] = []
        for i, (n, m) in enumerate(zip(names, blocks)):
            if freeze_backbone and i >= unfreeze_idx:
                _set_requires_grad(m, True)
                self._trainable_modules.append(m)
            else:
                if freeze_backbone:
                    self._frozen_modules.append(m)
        self.features = nn.Sequential(*blocks)
        self.hpool = nn.AdaptiveAvgPool2d((pool_h, None))
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            feat = self.features(dummy)
            feat = self.hpool(feat)
            _, c, h, w = feat.shape
            lstm_in = c * h
            self._seq_len = w
            self._lstm_in = lstm_in
        self.lstm = nn.LSTM(
            input_size=self._lstm_in,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        head_in = lstm_hidden * (2 if bidirectional else 1)
        self.classifier = nn.Linear(head_in, num_classes)
        self.AuxLogits = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for m in self._frozen_modules:
            if m.training:
                m.eval()
        feat = self.features(x)
        feat = self.hpool(feat)
        seq = feat.permute(0, 3, 2, 1).contiguous().view(x.size(0), -1, self._lstm_in)
        self.lstm.flatten_parameters()
        out, _ = self.lstm(seq)
        out = out.mean(dim=1)
        return self.classifier(out)

def create_model(input_shape, config):
    opts = {
        "freeze_backbone": config.get("freeze_backbone", False),
        "unfreeze_from": config.get("unfreeze_from", "Mixed_6e"),
        "pool_h": int(config.get("pool_h", 4)),
        "lstm_hidden": int(config.get("lstm_hidden", 512)),
        "lstm_layers": int(config.get("lstm_layers", 1)),
        "bidirectional": bool(config.get("bidirectional", True)),
        "transform_input": bool(config.get("transform_input", False)),
    }
    num_classes = int(config["num_classes"])
    model = InceptionCRNN(
        input_shape=input_shape,
        num_classes=num_classes,
        **opts,
    )
    return model
