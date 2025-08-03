import argparse
from collections import namedtuple
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from crnn_lid.manage.evaluate import evaluate
from sklearn.metrics import (
    f1_score as sk_f1,
    precision_score,
    recall_score,
    accuracy_score,
)
from torch.optim import Adam
from yaml import safe_load

from . import data_loaders, models


def _to_tensor(x: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(x).permute(0, 3, 1, 2).float().pin_memory().to(device, non_blocking=True)


def train(cli_args, log_dir: Path) -> Path:
    config_path = Path(cli_args.config)

    if not config_path.is_absolute() and not config_path.exists():
        config_path = Path(__file__).resolve().parent / cli_args.config

    with config_path.open() as f:
        cfg = safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    DataLoaderCls = getattr(data_loaders, cfg["data_loader"])
    train_gen = DataLoaderCls(cfg["train_data_dir"], cfg)
    val_gen = DataLoaderCls(cfg["validation_data_dir"], cfg)

    ModelCls = getattr(models, cfg["model"])
    ishape = train_gen.get_input_shape()
    model_shape = (ishape[2], ishape[0], ishape[1])
    net = ModelCls.create_model(model_shape, cfg).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = Adam(net.parameters(), lr=cfg["learning_rate"])

    best_val_f1 = 0.0
    best_path = log_dir / "best_model.pth"

    for epoch in range(1, cfg["num_epochs"] + 1):
        net.train()
        for xb, yb in train_gen.get_data(should_shuffle=True):
            xb_t = _to_tensor(xb, device)
            yb_t = torch.from_numpy(np.argmax(yb, axis=1)).long().to(device)

            optimizer.zero_grad()
            pred = net(xb_t)
            loss = criterion(pred, yb_t)
            loss.backward()
            optimizer.step()

        net.eval()
        with torch.no_grad():
            y_true, y_pred = [], []
            for xb, yb in val_gen.get_data(should_shuffle=False):
                xb_t = _to_tensor(xb, device)
                logits = net(xb_t)
                y_true.append(np.argmax(yb, axis=1))
                y_pred.append(logits.argmax(dim=1).cpu().numpy())
            y_true = np.concatenate(y_true)
            y_pred = np.concatenate(y_pred)

            val_f1 = sk_f1(y_true, y_pred, average="macro")
            val_acc = accuracy_score(y_true, y_pred)
            val_prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
            val_rec = recall_score(y_true, y_pred, average="macro", zero_division=0)

        print(
            f"Epoch {epoch}/{cfg['num_epochs']} | "
            f"acc={val_acc:.4f} prec={val_prec:.4f} rec={val_rec:.4f} f1={val_f1:.4f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(
                {
                    "state_dict": net.state_dict(),
                    "architecture": cfg["model"],
                    "input_shape": model_shape,
                    "num_classes": cfg["num_classes"],
                },
                best_path,
            )

    return best_path


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    default_cfg = (Path(__file__).resolve().parent / "config.yaml").as_posix()
    p.add_argument("--config", default=default_cfg)
    cli = p.parse_args()

    out = Path("C:/Users/fraca/Documents/GitHub/crnn_lid/manage/logs") / datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    out.mkdir(parents=True, exist_ok=True)

    best = train(cli, out)

    Dummy = namedtuple("Args", ["model_dir", "config", "use_test_set"])
    evaluate(Dummy(str(best), cli.config, False))
