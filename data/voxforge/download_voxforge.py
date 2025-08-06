import argparse
import os
import re
import shutil
import tarfile
import tempfile
import urllib.request
import wave
from typing import List, IO, cast

ROOT_DIR = r"C:\Users\fraca\Documents\GitHub\crnn_lid\data\voxforge"

LANG_URLS = {
    "german": "https://repository.voxforge1.org/downloads/de/Trunk/Audio/Main/16kHz_16bit/",
    "english": "https://repository.voxforge1.org/downloads/SpeechCorpus/Trunk/Audio/Main/16kHz_16bit/",
    "french": "https://repository.voxforge1.org/downloads/fr/Trunk/Audio/Main/16kHz_16bit/",
    "spanish": "https://repository.voxforge1.org/downloads/es/Trunk/Audio/Main/16kHz_16bit/",
    "italian": "https://repository.voxforge1.org/downloads/it/Trunk/Audio/Main/16kHz_16bit/",
}


def _copy_chunks(src: IO[bytes], dst: IO[bytes], size: int = 64 * 1024) -> None:
    while True:
        chunk = src.read(size)
        if not chunk:
            break
        dst.write(chunk)


def list_archives(base_url: str) -> List[str]:
    with urllib.request.urlopen(base_url) as resp:
        html = resp.read().decode()
    tgz_links = re.findall(r'href="([^"]+\.tgz)"', html, flags=re.IGNORECASE)
    return sorted(set(tgz_links))


def download(url: str, destination: str) -> None:
    with urllib.request.urlopen(url) as response, open(destination, "wb") as out:
        _copy_chunks(cast(IO[bytes], response), out)


def wav_duration(path: str) -> float:
    with wave.open(path, "rb") as w:
        return w.getnframes() / w.getframerate()


def process_archive(tgz_path: str, out_dir: str, prefix: str, min_seconds: float = 3.0) -> None:
    with tarfile.open(tgz_path, "r:gz") as tar:
        for member in tar.getmembers():
            if member.isfile() and member.name.lower().endswith(".wav"):
                extracted = tar.extractfile(member)
                if extracted is None:
                    continue
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav", mode="wb") as tmp:
                    _copy_chunks(cast(IO[bytes], extracted), tmp)
                    tmp_path = tmp.name
                try:
                    if wav_duration(tmp_path) > min_seconds:
                        target_name = f"{prefix}-{os.path.basename(member.name)}"
                        shutil.move(tmp_path, os.path.join(out_dir, target_name))
                finally:
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and extract VoxForge corpora without external tools.")
    parser.add_argument("language", choices=LANG_URLS.keys())
    parser.add_argument("--max", type=int, default=105, metavar="N")
    args = parser.parse_args()

    base_url = LANG_URLS[args.language]
    print(f"Getting index from {base_url} …")
    archives = list_archives(base_url)[: args.max]
    if not archives:
        print("No .tgz archives found – aborting.")
        return

    tmp_dir = os.path.join(ROOT_DIR, "tmp")
    out_dir = os.path.join(ROOT_DIR, args.language)
    os.makedirs(tmp_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    for idx, name in enumerate(archives, 1):
        url = base_url + name
        tgz_path = os.path.join(tmp_dir, name)
        print(f"[{idx}/{len(archives)}] Downloading {name}")
        download(url, tgz_path)
        print("  Extracting and filtering WAVs …")
        process_archive(tgz_path, out_dir, os.path.splitext(name)[0])
        os.remove(tgz_path)

    print("Done – audio saved in", out_dir)


if __name__ == "__main__":
    main()
