import itertools
import random
from pathlib import Path

LABELS = {
    "english": 0,
    "italian": 1,
    "sardinian": 2,
    "spanish": 3,
}


def recursive_glob(root: Path, pattern: str):
    for file in root.rglob(pattern):
        if file.is_file():
            yield file.resolve()


def get_immediate_subdirectories(path: Path):
    return [d for d in path.iterdir() if d.is_dir()]


def create_csv(
        root_dir: str | Path,
        train_validation_split: float = 0.8,
        train_csv_path: str | Path | None = None,
        val_csv_path: str | Path | None = None,
        test_csv_path: str | Path | None = None,
):
    root_dir = Path(root_dir)
    languages = [d.name for d in get_immediate_subdirectories(root_dir) if d.name in LABELS]

    counters: dict[str, int] = {}
    file_names: dict[str, list[Path]] = {}

    for lang in languages:
        files = list(recursive_glob(root_dir / lang, "*.wav"))
        files.extend(recursive_glob(root_dir / lang, "*.png"))
        counters[lang] = len(files)
        file_names[lang] = files

    smallest = min(counters.values())

    num_test = int(smallest * 0.1)
    num_train = int(smallest * (train_validation_split - 0.1))
    num_val = smallest - num_train - num_test

    train, val, test = [], [], []

    for lang in languages:
        label = LABELS[lang]
        train += zip(file_names[lang][:num_train], itertools.repeat(label))
        val += zip(file_names[lang][num_train: num_train + num_val], itertools.repeat(label))
        test += zip(file_names[lang][num_train + num_val: num_train + num_val + num_test], itertools.repeat(label))

    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)

    train_csv_path = Path(train_csv_path or root_dir / "training.csv")
    val_csv_path = Path(val_csv_path or root_dir / "validation.csv")
    test_csv_path = Path(test_csv_path or root_dir / "testing.csv")

    for p in (train_csv_path, val_csv_path, test_csv_path):
        p.parent.mkdir(parents=True, exist_ok=True)

    for dataset, path in ((train, train_csv_path), (val, val_csv_path), (test, test_csv_path)):
        with path.open("w") as f:
            for fname, lbl in dataset:
                f.write(f"{fname},{lbl}\n")

    print(f"[Training]   Files per language: {num_train} Total: {num_train * len(languages)}")
    print(f"[Validation] Files per language: {num_val} Total: {num_val * len(languages)}")
    print(f"[Testing]    Files per language: {num_test} Total: {num_test * len(languages)}")


if __name__ == "__main__":
    root = Path("C:/Users/fraca/Documents/GitHub/crnn_lid/data/spectrograms")
    create_csv(
        root,
        train_csv_path="C:/Users/fraca/Documents/GitHub/crnn_lid/train_data_dir/training.csv",
        val_csv_path="C:/Users/fraca/Documents/GitHub/crnn_lid/validation_data_dir/validation.csv",
        test_csv_path="C:/Users/fraca/Documents/GitHub/crnn_lid/test_data_dir/testing.csv",
    )
