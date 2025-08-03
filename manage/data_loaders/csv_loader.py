import abc
import csv
from pathlib import Path

import numpy as np
from crnn_lid.manage.utils import to_categorical


class CSVLoader(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, data_dir, config):

        self.config = config
        self.images_label_pairs = []
        self.input_shape = tuple(config["input_shape"])

        data_path = Path(data_dir)

        if not data_path.is_absolute():
            data_path = Path(__file__).resolve().parents[3] / data_path

        if not data_path.exists():
            raise FileNotFoundError(f"CSV file not found: {data_path}")

        with data_path.open("r", newline="") as csvfile:
            for (file_path, label) in list(csv.reader(csvfile)):
                self.images_label_pairs.append((file_path, int(label)))

    def get_data(self, should_shuffle=True, is_prediction=False, return_labels=False):
        data_pairs = self.images_label_pairs.copy()

        if should_shuffle:
            np.random.shuffle(data_pairs)

        total_files = len(data_pairs)
        batch_size = self.config["batch_size"]

        for start in range(0, total_files, batch_size):
            end = min(start + batch_size, total_files)
            current_batch = data_pairs[start:end]

            actual_batch_size = len(current_batch)
            data_batch = np.zeros((actual_batch_size,) + self.input_shape)
            label_batch = np.zeros((actual_batch_size, self.config["num_classes"]))

            for i, (file_path, label) in enumerate(current_batch):
                data = self.process_file(file_path)
                height, width, channels = data.shape
                data_batch[i, :height, :width, :] = data
                label_batch[i, :] = to_categorical([label], num_classes=self.config["num_classes"])

            if is_prediction:
                if return_labels:
                    yield data_batch, label_batch
                else:
                    yield (data_batch,)
            else:
                yield data_batch, label_batch

    def get_input_shape(self):
        return self.input_shape

    def get_num_files(self):
        return len(self.images_label_pairs)

    def get_num_batches(self):
        total_files = len(self.images_label_pairs)
        batch_size = self.config["batch_size"]
        return (total_files + batch_size - 1) // batch_size

    def get_labels(self):
        return [label for (file_path, label) in self.images_label_pairs]

    @abc.abstractmethod
    def process_file(self, file_path):

        raise NotImplementedError("Implement in child class.")
