import argparse
import pickle
import time
from math import ceil, sqrt
from pathlib import Path
from typing import List, Tuple

import imageio.v2 as imageio
import numpy as np
import torch
import yaml
from crnn_lid.manage.models.topcoder_crnn_finetune import create_model

OUTPUT_DIR = Path("C:/Users/fraca/Documents/GitHub/crnn_lid/img_conv_filter")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def deprocess_image(x: np.ndarray) -> np.ndarray:
    x = x.copy()
    x -= x.mean()
    std = x.std()
    if std < 1e-5:
        return np.zeros_like(x, dtype=np.uint8)
    x /= std
    x = np.tanh(x)
    x = (x + 1) / 2
    x = np.clip(x, 0, 1)
    x *= 255
    if x.shape[0] in (1, 3):
        x = np.transpose(x, (1, 2, 0))
    return x.astype("uint8")


def create_stitched_image(
        kept: List[Tuple[np.ndarray, float]], w: int, h: int, margin: int = 5
) -> np.ndarray:
    n = int(ceil(sqrt(len(kept))))
    missing = n * n - len(kept)
    kept += [(np.zeros((h, w, 1), dtype=np.uint8), 0.0)] * missing
    canvas = np.zeros((n * h + (n - 1) * margin, n * w + (n - 1) * margin, 1), np.uint8)
    for i in range(n):
        for j in range(n):
            img, _ = kept[i * n + j]
            y = i * (h + margin)
            x = j * (w + margin)
            canvas[y: y + h, x: x + w] = img
    return canvas


@torch.no_grad()
def _register_activation_hook(layer, store):
    def hook(_, __, output):
        store["activation"] = output

    return layer.register_forward_hook(hook)


def visualize_conv_filters(
        model: torch.nn.Module,
        layer: torch.nn.Module,
        n_filters: int,
        img_w: int,
        img_h: int,
        *,
        max_filters: int = 64,
        steps: int = 30,
        step_size: float = 1.0,
        device: torch.device,
        save_pickle: bool = False,
        load_pickle: bool = False,
) -> None:
    layer_name = layer.__class__.__name__
    pkl_file = OUTPUT_DIR / f"{layer_name}_filters.pkl"

    kept: List[Tuple[np.ndarray, float]] = []
    if load_pickle and pkl_file.exists():
        try:
            with pkl_file.open("rb") as f_read:
                kept = pickle.load(f_read)
            print(f"Loaded filters from {pkl_file}")
        except Exception as exc:  # noqa: BLE001
            print(f"Could not load {pkl_file}: {exc}")

    if not kept:
        act_store = {}
        hook = _register_activation_hook(layer, act_store)

        for idx in range(min(n_filters, max_filters)):
            print(f"Filter {idx}")
            start = time.time()
            inp = torch.randn(1, 1, img_h, img_w, device=device, requires_grad=True)
            optimizer = torch.optim.SGD([inp], lr=step_size)

            loss_val = -1.0
            for _ in range(steps):
                optimizer.zero_grad()
                model(inp)
                activation = act_store["activation"]
                loss = -activation[0, idx].mean()
                loss.backward()
                inp.grad.data /= inp.grad.data.std().clamp_min(1e-8)
                optimizer.step()
                loss_val = -loss.item()

            if loss_val > 0:
                img = deprocess_image(inp.detach().cpu().numpy()[0])
                kept.append((img, loss_val))
            print(f"done in {int(time.time() - start)} s")

        hook.remove()
        if save_pickle:
            pkl_file.write_bytes(pickle.dumps(kept))
            print(f"Saved pickle to {pkl_file}")

    kept.sort(key=lambda t: t[1], reverse=True)
    kept = kept[:max_filters]
    stitched = create_stitched_image(kept, img_w, img_h)
    out_name = OUTPUT_DIR / f"{layer_name}_{int(sqrt(max_filters))}x{int(sqrt(max_filters))}.png"
    imageio.imwrite(out_name, stitched.squeeze())
    print(f"Saved {out_name}")


def visualize_conv_layers(cli_args, cfg: dict) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    img_w = cli_args.width or cfg["input_shape"][1]
    img_h = cli_args.height or cfg["input_shape"][0]

    input_shape = (cfg["input_shape"][2], cfg["input_shape"][0], cfg["input_shape"][1])
    model = create_model(input_shape, cfg).to(device)
    model.eval()

    _ = model(torch.randn(1, 1, img_h, img_w, device=device))

    for layer in model.modules():
        if isinstance(layer, torch.nn.Conv2d):
            print(f"{layer.__class__.__name__} – {layer.out_channels} filters")
            visualize_conv_filters(
                model,
                layer,
                layer.out_channels,
                img_w,
                img_h,
                max_filters=cli_args.num_filter,
                steps=20,
                step_size=1.0,
                device=device,
                save_pickle=cli_args.save_pickle,
                load_pickle=cli_args.load_pickle,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--num-filter", type=int, default=64)
    parser.add_argument("--save-pickle", action="store_true")
    parser.add_argument("--load-pickle", action="store_true")
    args = parser.parse_args()

    conf_path = Path(args.config)
    if not conf_path.is_absolute():
        conf_path = Path(__file__).resolve().parent / conf_path
    if not conf_path.exists():
        raise FileNotFoundError(f"Config file not found: {conf_path}")

    with conf_path.open("r", encoding="utf-8") as cfg_file:
        cfg_dict = yaml.safe_load(cfg_file)

    visualize_conv_layers(args, cfg_dict)
