import os
import argparse
import imageio
import numpy as np
import sys

lib_dir = os.path.join(os.getcwd(), "../keras/data_loaders")
sys.path.append(lib_dir)

from SpectrogramGenerator import SpectrogramGenerator
from NoisyBackgroundSpectrogramGenerator import NoisyBackgroundSpectrogramGenerator
from VinylBackgroundSpectrogramGenerator import VinylBackgroundSpectrogramGenerator
from MusicBackgroundSpectrogramGenerator import MusicBackgroundSpectrogramGenerator
from create_csv import create_csv

def directory_to_spectrograms(args):

    source = args.source
    config = {
        "pixel_per_second": args.pixel_per_second,
        "input_shape": args.shape
    }

    # Start a spectrogram generator for each class
    # Each generator will scan a directory for audio files and convert them to spectrogram images
    # adjust this if you have other languages or any language is missing
    languages = ["english",
                 "german"]

    generators = [SpectrogramGenerator(os.path.join(source, language), config, shuffle=False, run_only_once=True) for language in languages]

    for language, generator in zip(languages, generators):
        print(f"Files trovati per '{language}':")
        for f in generator.files:
            print(f)

    generator_queues = [SpectrogramGen.get_generator() for SpectrogramGen in generators]

    for language in languages:
        output_dir = os.path.join(args.target, language)
        os.makedirs(output_dir, exist_ok=True)

    i = 0
    while True:

        target_shape = tuple(args.shape)

        try:
            for j, language in enumerate(languages):

                data = next(generator_queues[j])

                assert data.shape == target_shape, "Shape mismatch {} vs {}".format(data.shape, args.shape)

                file_name = os.path.join(args.target, language, "{}.png".format(i))
                imageio.imwrite(file_name, np.squeeze(data))

            i += 1

            if i % 1000 == 0:
                print("Processed {} images".format(i))

        except StopIteration:
            print("Saved {} images. Stopped on {}".format(i, language))
            break


if __name__ == "__main__":

    source_dir = "/mnt/c/Users/fraca/Documents/GitHub/crnn-lid/data/voxforge"
    target_dir = "/mnt/c/Users/fraca/Documents/GitHub/crnn-lid/data/spectrograms"

    class Args:
        pass

    cli_args = Args()
    cli_args.source = source_dir
    cli_args.target = target_dir
    cli_args.shape = [129, 200, 1]
    cli_args.pixel_per_second = 50

    directory_to_spectrograms(cli_args)
    
    train_csv_path = "/mnt/c/Users/fraca/Documents/GitHub/crnn-lid/train_data_dir/training.csv"
    val_csv_path = "/mnt/c/Users/fraca/Documents/GitHub/crnn-lid/validation_data_dir/validation.csv"
    test_csv_path = "/mnt/c/Users/fraca/Documents/GitHub/crnn-lid/test_data_dir/testing.csv"

    create_csv(target_dir, train_validation_split=0.8,
               train_csv_path=train_csv_path,
               val_csv_path=val_csv_path,
               test_csv_path=test_csv_path)