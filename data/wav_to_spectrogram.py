import argparse
from pathlib import Path

import imageio.v3 as imageio
import numpy as np

from crnn_lid.manage.data_loaders.SpectrogramGenerator import SpectrogramGenerator
from crnn_lid.data.create_csv import create_csv


def directory_to_spectrograms(
        source: Path,
        target: Path,
        shape: list[int],
        pps: int,
        segment_length: float | None = None
):
    config: dict[str, int | float | list[int]] = {
        "pixel_per_second": pps,
        "input_shape": shape
    }
    if segment_length is not None:
        config["segment_length"] = segment_length

    languages = ["english", "italian", "spanish", "sardinian"]

    generators = [
        SpectrogramGenerator(source / lang, config, shuffle=False, run_only_once=True)
        for lang in languages
    ]
    iterators = [iter(g) for g in generators]

    for lang in languages:
        (target / lang).mkdir(parents=True, exist_ok=True)

    idx = 0
    while True:
        try:
            segments = [next(it) for it in iterators]
        except StopIteration:
            break

        for lang, seg in zip(languages, segments):
            assert seg.shape == tuple(shape), f"Shape mismatch {seg.shape} vs {shape}"
            imageio.imwrite(target / lang / f"{idx}.png", np.squeeze(seg))

        idx += 1
        if idx % 1000 == 0:
            print(f"Processed {idx} images")

    print(f"Saved {idx} images.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        default=r"C:\Users\fraca\Documents\GitHub\crnn_lid\data\voxforge"
    )
    parser.add_argument(
        "--target",
        default=r"C:\Users\fraca\Documents\GitHub\crnn_lid\data\spectrograms"
    )
    parser.add_argument(
        "--shape",
        nargs=3,
        type=int,
        default=[129, 200, 1]
    )
    parser.add_argument(
        "--pixel-per-second",
        dest="pps",
        type=int,
        default=50
    )
    parser.add_argument(
        "--segment-length",
        dest="segment_length",
        type=float,
        default=None
    )
    args = parser.parse_args()

    directory_to_spectrograms(
        Path(args.source),
        Path(args.target),
        args.shape,
        args.pps,
        args.segment_length
    )

    create_csv(
        Path(args.target),
        train_csv_path=Path("train_data_dir") / "training.csv",
        val_csv_path=Path("validation_data_dir") / "validation.csv",
        test_csv_path=Path("test_data_dir") / "testing.csv",
    )
