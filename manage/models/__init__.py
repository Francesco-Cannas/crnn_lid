import importlib
import os

import torch

from .cnn import CNN
from .crnn import CRNN
from .inceptionv3 import create_model
from .inceptionv3_crnn import InceptionCRNN
from .resnet import create_model
from .topcoder import create_model
from .topcoder_5s_finetune import create_model
from .topcoder_crnn import TopcoderCRNN
from .topcoder_crnn_finetune import TopcoderCRNNFinetune
from .topcoder_deeper import create_model
from .topcoder_finetune import create_model
from .topcoder_small import TopcoderCNNSmall
from .xception import create_model


def _module_from_name(name: str):
    return importlib.import_module(f".{name}", package=__name__)


def load_model(weights_path: str | os.PathLike, device: str | torch.device | None = None) -> torch.nn.Module:
    ckpt = torch.load(weights_path, map_location="cpu")
    arch = ckpt["architecture"]
    input_shape = tuple(ckpt["input_shape"])
    module = _module_from_name(arch)
    model = module.create_model(input_shape, {"num_classes": ckpt["num_classes"]})
    model.load_state_dict(ckpt["state_dict"])
    model.to(device or ("cuda" if torch.cuda.is_available() else "cpu")).eval()
    return model
