import numpy as np
import imageio
from .csv_loader import CSVLoader

class ImageLoader(CSVLoader):

    def process_file(self, file_path):
        
        if self.config.get("color_mode", "RGB") == "L":
            img = Image.open(file_path).convert('L')
            img_array = np.array(img).astype(np.float32) / 255.0
            img_array = np.expand_dims(img_array, axis=-1)  # (H, W, 1)
        else:
            img = Image.open(file_path).convert('RGB')
            img_array = np.array(img).astype(np.float32) / 255.0
        
        return img_array