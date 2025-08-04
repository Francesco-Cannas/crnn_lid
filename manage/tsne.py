import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from crnn_lid.manage import data_loaders, models
from pandas import DataFrame
from sklearn.manifold import TSNE

matplotlib.use("pdf")

OUTPUT_DIR = Path(r"C:/Users/fraca/Documents/GitHub/crnn_lid/manage").resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _plot_with_labels(low_d, labels, label_names, out_path: Path) -> None:
    df = DataFrame({"x": low_d[:, 0], "y": low_d[:, 1], "label": labels})
    groups = df.groupby("label")

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.margins(0.05)
    uniq = sorted(set(labels))
    cmap = plt.get_cmap("tab10", len(uniq))

    for idx, lab in enumerate(uniq):
        grp = groups.get_group(lab)
        ax.scatter(
            grp.x,
            grp.y,
            color=cmap(idx),
            label=label_names[lab],
            alpha=0.7,
            s=60,
            edgecolors="k",
            linewidth=0.5,
        )
        cx, cy = grp.x.mean(), grp.y.mean()
        ax.text(
            cx,
            cy,
            label_names[lab],
            fontsize=12,
            weight="bold",
            ha="center",
            va="center",
            bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"),
        )

    ax.legend(title="Classes", fontsize=9)
    ax.set_title("t-SNE visualization of model outputs")
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    fig.tight_layout()
    fig.savefig(out_path)
    fig.savefig(out_path.with_suffix(".png"))
    plt.close(fig)


def visualize_cluster(args) -> None:
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = Path(__file__).resolve().parent / cfg_path
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with cfg_path.open() as f:
        cfg = yaml.safe_load(f)

    DataLoader = getattr(data_loaders, cfg["data_loader"])
    gen = DataLoader(cfg["validation_data_dir"], cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model:
        ckpt_path = Path(args.model)
    elif cfg.get("model_checkpoint"):
        ckpt_path = Path(cfg["model_checkpoint"])
    else:
        logs_dir = Path(__file__).resolve().parents[2] / "logs"
        ckpts = sorted(logs_dir.glob("*/best_model.pth"))
        if not ckpts:
            raise FileNotFoundError("Nessun checkpoint trovato in logs/*/best_model.pth")
        ckpt_path = ckpts[-1]

    print("Checkpoint usato:", ckpt_path)
    model = models.load_model(str(ckpt_path)).to(device)
    model.eval()

    all_probs, all_labels, all_files = [], [], []
    with torch.no_grad():
        offset = 0
        for xb, yb in gen.get_data(
                should_shuffle=False, is_prediction=False, return_labels=True
        ):
            logits = model(
                torch.from_numpy(xb)
                .permute(0, 3, 1, 2)
                .float()
                .to(device, non_blocking=True)
            )
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(np.argmax(yb, axis=1))

            batch_files = [
                fp for fp, _ in gen.images_label_pairs[offset: offset + len(xb)]
            ]
            all_files.extend(batch_files)
            offset += len(xb)

    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)

    num_samples = min(args.limit, probs.shape[0])
    probs = probs[:num_samples]
    labels = labels[:num_samples]
    files_subset = all_files[:num_samples]
    print(f"Using {num_samples} samples for t-SNE")

    plot_path = OUTPUT_DIR / args.plot
    np.savetxt(OUTPUT_DIR / "probabilities.tsv", probs, delimiter="\t")
    DataFrame({"label": labels, "filename": files_subset}).to_csv(
        OUTPUT_DIR / "metadata.tsv", sep="\t", index=False
    )

    tsne = TSNE(
        perplexity=30,
        n_components=2,
        init="pca",
        max_iter=args.iter,
        random_state=42,
        learning_rate="auto",
    )
    low_d = tsne.fit_transform(probs)
    _plot_with_labels(low_d, labels, cfg["label_names"], plot_path)
    print("Plot salvato in:", plot_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="Checkpoint .pth del modello")
    parser.add_argument("--config", required=True, help="Path al config.yaml")
    parser.add_argument(
        "--plot",
        default="tsne.pdf",
        help="Nome file plot (sarà salvato in OUTPUT_DIR)",
    )
    parser.add_argument("--limit", type=int, default=2000, help="Max campioni")
    parser.add_argument("--iter", type=int, default=4000, help="Iterazioni t-SNE")
    args = parser.parse_args()

    if args.model is None:
        candidate_roots = [
            Path.cwd() / "C:/Users/fraca/Documents/GitHub/crnn_lid/manage/logs",
            Path(__file__).resolve().parents[3] / "logs",
        ]
        ckpts = []
        for root in candidate_roots:
            ckpts.extend(root.glob("*/best_model.pth"))
        if not ckpts:
            raise FileNotFoundError("Nessun best_model.pth trovato in logs/")
        args.model = str(sorted(ckpts)[-1])
        print("Checkpoint auto-selezionato:", args.model)

    visualize_cluster(args)
