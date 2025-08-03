import argparse
import csv

import imageio.v3 as imageio
import numpy as np


def main(args):
    with open(args.csv_input, "r", newline='') as csv_file:
        reader = csv.reader(csv_file, delimiter=",")
        for row in reader:
            image_path, label = row
            check_image(image_path)

def check_image(image_path):
    try:
        image = imageio.imread(image_path)
    except Exception as e:
        print(f"Errore nel leggere l'immagine {image_path}: {e}")
        return
    print(f"Controllo immagine: {image_path}, shape: {image.shape}")
    if image.ndim == 3:
        image = np.mean(image, axis=2)
    mean_val = np.mean(image)
    diff = image - mean_val
    nonzero_count = np.count_nonzero(diff)
    print(f"Mean pixel value: {mean_val}, nonzero diff count: {nonzero_count}")
    if nonzero_count == 0:
        print(f"Immagine uniforme trovata: {image_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', dest='csv_input', required=True,
                        help='Path to the csv file containing paths and labels of images')
    args = parser.parse_args()
    main(args)