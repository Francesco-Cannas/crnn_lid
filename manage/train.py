import argparse
import itertools
import sys
from collections import namedtuple
from datetime import datetime
from io import StringIO
from pathlib import Path
from time import perf_counter
from typing import List

import numpy as np
import torch
from crnn_lid.manage.evaluate import evaluate
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import (
    f1_score as sk_f1,
    precision_score,
    recall_score,
    accuracy_score,
    confusion_matrix,
)
from torch.optim import Adam
from yaml import safe_load

from . import data_loaders, models


def _to_tensor(x: np.ndarray, device: torch.device) -> torch.Tensor:
    return (
        torch.from_numpy(x)
        .permute(0, 3, 1, 2)
        .float()
        .pin_memory()
        .to(device, non_blocking=True)
    )


def _epoch_train(model, data_loader, criterion, optimizer, device):
    model.train()
    start = perf_counter()
    total_loss = 0.0
    num_samples = 0
    num_steps = 0

    for xb, yb in data_loader.get_data(should_shuffle=True):
        xb_t = _to_tensor(xb, device)
        yb_t = torch.from_numpy(np.argmax(yb, axis=1)).long().to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(xb_t)
        loss = criterion(logits, yb_t)
        loss.backward()
        optimizer.step()
        bs = xb_t.size(0)
        total_loss += loss.item() * bs
        num_samples += bs
        num_steps += 1

    runtime = perf_counter() - start
    avg_loss = total_loss / max(1, num_samples)
    samples_per_sec = num_samples / runtime if runtime > 0 else 0.0
    steps_per_sec = num_steps / runtime if runtime > 0 else 0.0
    return avg_loss, runtime, samples_per_sec, steps_per_sec, num_samples, num_steps


def _render_text_page(pdf: PdfPages, lines: List[str]):
    fig = plt.figure(figsize=(8.27, 11.69), facecolor="white")
    ax = fig.add_subplot(111)
    ax.axis("off")
    y = 0.98

    for line in lines:
        ax.text(0.01, y, line, va="top", ha="left", fontsize=10, family="monospace")
        y -= 0.03

        if y < 0.02:
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            fig = plt.figure(figsize=(8.27, 11.69), facecolor="white")
            ax = fig.add_subplot(111)
            ax.axis("off")
            y = 0.98
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _plot_confusion(cm: np.ndarray, num_classes: int, pdf: PdfPages, png_path: Path):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_title("Confusion Matrix")
    tick_marks = np.arange(num_classes)
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    thresh = cm.max() / 2.0 if cm.max() else 0.0

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(
            j,
            i,
            format(cm[i, j], "d"),
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=8,
            color="white" if cm[i, j] > thresh else "black",
        )

    fig.tight_layout()
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    pdf.savefig(fig)
    plt.close(fig)


def train(cli_args, log_dir: Path) -> Path:
    config_path = Path(cli_args.config)

    if not config_path.is_absolute() and not config_path.exists():
        config_path = Path(__file__).resolve().parent / cli_args.config
    with open(config_path, "r") as f:
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
    total_runtime = 0.0
    total_loss_weighted = 0.0
    total_samples = 0
    total_steps = 0

    log_lines: List[str] = []
    pdf_lines: List[str] = []
    output_dir = Path("C:/Users/fraca/Documents/GitHub/crnn_lid/manage")
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / "training_report.pdf"

    pdf = PdfPages(pdf_path.as_posix())

    for epoch in range(1, cfg["num_epochs"] + 1):
        (
            train_loss,
            train_runtime,
            train_sps,
            train_stepsps,
            epoch_samples,
            epoch_steps,
        ) = _epoch_train(net, train_gen, criterion, optimizer, device)

        total_runtime += train_runtime
        total_loss_weighted += train_loss * epoch_samples
        total_samples += epoch_samples
        total_steps += epoch_steps

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

        epoch_line = (
            f"Epoch {epoch}/{cfg['num_epochs']} | "
            f"Train_loss={train_loss:.4f} "
            f"Train_runtime={train_runtime:.2f}s "
            f"Train_samples_per_second={train_sps:.2f} "
            f"Train_steps_per_second={train_stepsps:.2f} | "
            f"Val_acc={val_acc:.4f} "
            f"Val_prec={val_prec:.4f} "
            f"Val_rec={val_rec:.4f} "
            f"Val_f1={val_f1:.4f}"
        )

        print(epoch_line)
        log_lines.append(epoch_line)

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

    avg_train_loss = total_loss_weighted / max(1, total_samples)
    global_samples_per_sec = total_samples / total_runtime if total_runtime > 0 else 0.0
    global_steps_per_sec = total_steps / total_runtime if total_runtime > 0 else 0.0

    final_line = (
        "Final | "
        f"Train_runtime_total={total_runtime:.2f}s "
        f"Train_loss_avg={avg_train_loss:.4f} "
        f"Train_samples_per_second={global_samples_per_sec:.2f} "
        f"Train_steps_per_second={global_steps_per_sec:.2f}"
    )

    print(final_line)

    pdf_lines.append(final_line)
    buffer = StringIO()
    _stdout = sys.stdout
    sys.stdout = buffer
    Dummy = namedtuple("Args", ["model_dir", "config", "use_test_set"])
    y_true_eval, y_pred_eval = evaluate(Dummy(str(best_path), cli_args.config, False), return_data=True)

    sys.stdout = _stdout
    eval_text = buffer.getvalue().strip().splitlines()
    pdf_lines.extend(eval_text)
    cm = confusion_matrix(y_true_eval, y_pred_eval)
    cm_png_path = output_dir / "confusion_matrix.png"
    _plot_confusion(cm, cfg["num_classes"], pdf, cm_png_path)
    _render_text_page(pdf, pdf_lines)
    pdf.close()
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
