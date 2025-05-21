import numpy as np
import csv
import abc
from keras.utils import to_categorical

class CSVLoader(object):

    __metaclass__ = abc.ABCMeta

    def __init__(self, data_dir, config):

        self.config = config
        self.images_label_pairs = []
        self.input_shape = tuple(config["input_shape"])

        with open(data_dir, "r") as csvfile:
            for (file_path, label)in list(csv.reader(csvfile)):
                self.images_label_pairs.append((file_path, int(label)))

    def get_data(self, should_shuffle=True, is_prediction=False, return_labels=False):

        start = 0

        while True:
            data_batch = np.zeros((self.config["batch_size"], ) + self.input_shape)
            label_batch = np.zeros((self.config["batch_size"], self.config["num_classes"]))

            for i, (file_path, label) in enumerate(self.images_label_pairs[start:start + self.config["batch_size"]]):
                data = self.process_file(file_path)
                height, width, channels = data.shape
                data_batch[i, :height, :width, :] = data
                label_batch[i, :] = to_categorical([label], num_classes=self.config["num_classes"])

            start += self.config["batch_size"]

            if start + self.config["batch_size"] > self.get_num_files():
                start = 0
                if should_shuffle:
                    np.random.shuffle(self.images_label_pairs)

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
        return len(self.images_label_pairs) // self.config["batch_size"]
    
    def get_labels(self):
        return [label for (file_path, label) in self.images_label_pairs]

    @abc.abstractmethod
    def process_file(self, file_path):

        raise NotImplementedError("Implement in child class.")