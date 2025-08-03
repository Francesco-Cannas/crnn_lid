import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from crnn_lid.manage.data_loaders.SpectrogramGenerator import SpectrogramGenerator
from crnn_lid.manage.models import load_model
from yaml import safe_load


def predict(model_path: str, audio_path: str, cfg_path: str):
    cfg_file = Path(cfg_path)
    if not cfg_file.is_absolute():
        cfg_file = (Path(__file__).resolve().parent / cfg_file).resolve()
    if not cfg_file.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_file}")

    with cfg_file.open() as f:
        cfg = safe_load(f)

    gen = SpectrogramGenerator(audio_path, cfg, run_only_once=True)
    data = np.stack(list(gen))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, device)

    with torch.no_grad():
        logits = model(torch.from_numpy(data).permute(0, 3, 1, 2).float().to(device))
        probs = torch.softmax(logits, dim=1).cpu().numpy()

    classes = np.argmax(probs, axis=1)
    majority = int(np.bincount(classes).argmax())
    top3 = np.argsort(probs.mean(axis=0))[::-1][:3]

    print("segment_classes:", classes)
    print("majority_class:", cfg["label_names"][majority])
    print("confidence:", float(np.mean(probs[:, majority])))
    print("top3:", [(cfg["label_names"][i], float(probs[:, i].mean())) for i in top3])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    if not Path(args.input).is_file():
        sys.exit("Input is not a file.")

    predict(args.model, args.input, args.config)
