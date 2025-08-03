import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from yaml import safe_load

from . import data_loaders, models


def evaluate(cli_args):
    config_path = Path(cli_args.config)

    if not config_path.is_absolute() and not config_path.exists():
        config_path = Path(__file__).resolve().parent / cli_args.config

    with config_path.open() as f:
        cfg = safe_load(f)

    DataLoaderCls = getattr(data_loaders, cfg["data_loader"])
    loader = DataLoaderCls(
        cfg["test_data_dir"] if cli_args.use_test_set else cfg["validation_data_dir"], cfg
    )

    model = models.load_model(cli_args.model_dir)

    y_true, y_pred = [], []
    with torch.no_grad():
        for xb, yb in loader.get_data(should_shuffle=False):
            logits = model(
                torch.from_numpy(xb)
                .permute(0, 3, 1, 2)
                .float()
                .to(next(model.parameters()).device)
            )
            y_true.append(np.argmax(yb, axis=1))
            y_pred.append(logits.argmax(dim=1).cpu().numpy())

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Macro Precision:", precision_score(y_true, y_pred, average="macro", zero_division=0))
    print("Macro Recall:", recall_score(y_true, y_pred, average="macro", zero_division=0))
    print("Macro F1:", f1_score(y_true, y_pred, average="macro"))
    print(classification_report(y_true, y_pred, target_names=cfg["label_names"]))
    print("Confusion matrix:\n", confusion_matrix(y_true, y_pred))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", dest="model_dir", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--testset", dest="use_test_set", action="store_true")
    evaluate(ap.parse_args())
