import numpy as np
import imageio
from .csv_loader import CSVLoader

class ImageLoader(CSVLoader):

    def process_file(self, file_path):
        image = imageio.imread(file_path)

        if self.config.get("color_mode", "RGB") == "L" and len(image.shape) == 3:
            image = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
            image = image.astype(np.uint8)

        if len(image.shape) == 2:
            image = np.expand_dims(image, -1)

        assert len(image.shape) == 3

        return image / 255.0  # Normalizza a [0,1]