import os
import argparse
import fnmatch
import math
import itertools
from random import shuffle

LABELS = {
    "english": 0,
    "italian": 1,
    "spanish": 2,
    "sardinian": 3
}

def recursive_glob(path, pattern):
    for root, dirs, files in os.walk(path):
        for basename in files:
            if fnmatch.fnmatch(basename, pattern):
                filename = os.path.abspath(os.path.join(root, basename))
                if os.path.isfile(filename):
                    yield filename


def get_immediate_subdirectories(path):
    return [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]


def create_csv(root_dir, train_validation_split=0.8, train_csv_path=None, val_csv_path=None, test_csv_path=None):
    languages = [lang for lang in get_immediate_subdirectories(root_dir) if lang in LABELS]
    counter = {}
    file_names = {}

    # Count all files for each language
    for lang in languages:
        print(lang)
        files = list(recursive_glob(os.path.join(root_dir, lang), "*.wav"))
        files.extend(recursive_glob(os.path.join(root_dir, lang), "*.png"))
        num_files = len(files)

        file_names[lang] = files
        counter[lang] = num_files

    # Calculate train / validation split
    print(counter)
    smallest_count = min(counter.values())

    num_test = int(smallest_count * 0.1)
    num_train = int(smallest_count * (train_validation_split - 0.1))
    num_validation = smallest_count - num_train - num_test

    # Split datasets and shuffle languages
    training_set = []
    validation_set = []
    test_set = []

    for lang in languages:
        label = LABELS[lang]
        training_set += zip(file_names[lang][:num_train], itertools.repeat(label))
        validation_set += zip(file_names[lang][num_train:num_train + num_validation], itertools.repeat(label))
        test_set += zip(file_names[lang][num_train + num_validation:num_train + num_validation + num_test], itertools.repeat(label))

    shuffle(training_set)
    shuffle(validation_set)
    shuffle(test_set)

    if train_csv_path is None:
        train_csv_path = os.path.join(root_dir, "training.csv")
    if val_csv_path is None:
        val_csv_path = os.path.join(root_dir, "validation.csv")
    if test_csv_path is None:
        test_csv_path = os.path.join(root_dir, "testing.csv")

    os.makedirs(os.path.dirname(train_csv_path), exist_ok=True)
    os.makedirs(os.path.dirname(val_csv_path), exist_ok=True)
    os.makedirs(os.path.dirname(test_csv_path), exist_ok=True)

    with open(train_csv_path, "w") as train_file:
        for (filename, label) in training_set:
            train_file.write(f"{filename}, {label}\n")

    with open(val_csv_path, "w") as val_file:
        for (filename, label) in validation_set:
            val_file.write(f"{filename}, {label}\n")

    with open(test_csv_path, "w") as test_file:
        for (filename, label) in test_set:
            test_file.write(f"{filename}, {label}\n")

    # Stats
    print("[Training]   Files per language: {} Total: {}".format(num_train, num_train * len(languages)))
    print("[Validation] Files per language: {} Total: {}".format(num_validation, num_validation * len(languages)))
    print("[Testing]    Files per language: {} Total: {}".format(num_test, num_test * len(languages)))


if __name__ == '__main__':

    root_dir = "/mnt/c/Users/fraca/Documents/GitHub/crnn-lid/data/spectrograms"

    train_data_dir = "/mnt/c/Users/fraca/Documents/GitHub/crnn-lid/train_data_dir/training.csv"
    validation_data_dir = "/mnt/c/Users/fraca/Documents/GitHub/crnn-lid/validation_data_dir/validation.csv"
    test_data_dir = "/mnt/c/Users/fraca/Documents/GitHub/crnn-lid/test_data_dir/testing.csv"

    create_csv(root_dir, train_validation_split=0.8, train_csv_path=train_data_dir, val_csv_path=validation_data_dir, test_csv_path=test_data_dir)