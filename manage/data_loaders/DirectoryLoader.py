import os
import random
from itertools import cycle

import numpy as np

from crnn_lid.manage.data_loaders.SpectrogramGenerator import SpectrogramGenerator
from crnn_lid.manage.utils import to_categorical


class DirectoryLoader(object):

    def __init__(self, source_directory, config, shuffle=True):
        self.config = config
        self.source_directory = source_directory
        self.shuffle = shuffle

        self.generators = [
            SpectrogramGenerator(os.path.join(self.source_directory, "english"), config, shuffle=shuffle),
            SpectrogramGenerator(os.path.join(self.source_directory, "italian"), config, shuffle=shuffle),
            SpectrogramGenerator(os.path.join(self.source_directory, "spanish"), config, shuffle=shuffle),
            SpectrogramGenerator(os.path.join(self.source_directory, "sardinian"), config, shuffle=shuffle)
        ]

        self.generator_queues = [SpectrogramGen.get_generator() for SpectrogramGen in self.generators]

    def get_data(self):

        config = self.config

        while True:

            num_classes = len(self.generators)
            if self.shuffle:
                sample_selection = [random.randint(0, num_classes - 1) for _ in range(config["batch_size"])]
            else:
                label_sequence = cycle(range(num_classes))
                sample_selection = [next(label_sequence) for _ in range(config["batch_size"])]

            data_batch = np.zeros((config["batch_size"],) + tuple(config["input_shape"]))
            label_batch = np.zeros((config["batch_size"], config["num_classes"]))

            for i, label in enumerate(sample_selection):
                data = next(self.generator_queues[label])
                data = np.divide(data, 255.0)

                height, width, channels = data.shape
                data_batch[i, :height, :width, :] = data
                label_batch[i, :] = to_categorical([label], num_classes=config["num_classes"])

            yield data_batch, label_batch

    def get_num_files(self):
        min_num_files = min(x.get_num_files() for x in self.generators)
        return len(self.generators) * min_num_files * self.config["batch_size"]


if __name__ == "__main__":
    a = DirectoryLoader("C:/Users/fraca/Documents/GitHub/crnn_lid/data/spectrograms",
                        {"pixel_per_second": 50, "input_shape": [129, 200, 1], "batch_size": 32, "num_classes": 4},
                        shuffle=True)
    print(a.get_num_files())

    import imageio

    for data, labels in a.get_data():
        i = 0
        for image in np.vsplit(data, 32):
            imageio.imwrite(f"C:/Users/fraca/Documents/GitHub/crnn_lid/data/spectrograms/png/{i}.png",
                            np.squeeze(image) * 255.)
            i += 1
        break  # evita loop infinito nel test
