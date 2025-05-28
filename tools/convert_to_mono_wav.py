import os
import subprocess
import argparse

filetypes_to_convert = [".mp3", ".m4a", ".webm"]

def convert(filename):
    filename_extensionless, extension = os.path.splitext(filename)
    new_filename = filename_extensionless + ".wav"
    if not os.path.exists(new_filename):
        command = [
            "ffmpeg",
            "-i", filename,
            "-ac", "1",
            new_filename
        ]
        subprocess.run(command, check=True)

def walk_path(path):
    for root, dirs, files in os.walk(path):
        for sound_file in files:
            _, extension = os.path.splitext(sound_file)
            if extension.lower() in filetypes_to_convert:
                yield os.path.join(root, sound_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', dest='path', help='Directory for the files to convert', required=True)
    args = parser.parse_args()

    for sound_file in walk_path(args.path):
        convert(sound_file)