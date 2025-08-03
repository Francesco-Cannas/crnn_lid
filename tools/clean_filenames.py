import argparse
import os
import re
import shutil


def clean(filename):
    cleaned = re.sub(r"[^a-zA-Z0-9._\- ]", "", filename)
    cleaned = cleaned.replace("'", "")
    return re.sub(r"''{1,+}", "_", cleaned)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', dest='target', help='Directory for the filenames to be cleaned', required=True)
    args = parser.parse_args()

    os.chdir(args.target)

    for root, dirs, files in os.walk("."):
        for filename in files:
            new_filename = clean(filename)
            new_filepath = os.path.join(root, new_filename)
            old_filepath = os.path.join(root, filename)

            if new_filepath != old_filepath:
                if os.path.exists(new_filepath):
                    print(f"⚠️ File esistente: {new_filepath}, salta.")
                else:
                    print(f"{old_filepath} -> {new_filepath}")
                    shutil.move(old_filepath, new_filepath)