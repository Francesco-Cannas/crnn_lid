import random
from pathlib import Path
from queue import Queue
from typing import Generator, List

import librosa
import librosa.feature as lf
import numpy as np


def recursive_glob(root: str | Path, pattern: str) -> Generator[str, None, None]:
    root = Path(root)
    for file in root.rglob(pattern):
        if file.is_file():
            yield str(file.resolve())


class SpectrogramGenerator:
    def __init__(
            self,
            source: str | Path,
            config: dict,
            *,
            shuffle: bool = False,
            max_size: int = 100,
            run_only_once: bool = False,
    ) -> None:
        self.source = Path(source)
        self.config = config
        self.queue: "Queue[np.ndarray]" = Queue(max_size)
        self.shuffle = shuffle
        self.run_only_once = run_only_once

        if self.source.is_dir():
            files: List[str] = []
            files.extend(recursive_glob(self.source, "*.wav"))
            files.extend(recursive_glob(self.source, "*.mp3"))
            files.extend(recursive_glob(self.source, "*.m4a"))
        else:
            files = [str(self.source)]

        if self.shuffle:
            random.shuffle(files)
        self.files = files

    @staticmethod
    def _audio_to_spectrogram(
            file: str,
            pixel_per_sec: int,
            height: int,
            segment_length: float | None = None,
    ) -> np.ndarray:
        y, sr = librosa.load(file, sr=None)

        if segment_length is not None:
            max_len = int(sr * segment_length)
            if y.shape[0] > max_len:
                y = y[:max_len]
            else:
                y = np.pad(y, (0, max_len - y.shape[0]), mode="constant")

        n_fft = min(1024, max(256, int(sr * 0.025)))
        hop_length = n_fft // 4

        S = lf.melspectrogram(
            y=y,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=height,
            power=2.0,
        )

        S = librosa.power_to_db(S, ref=np.max)

        time_frames = max(1, int(np.ceil(len(y) / sr * pixel_per_sec)))
        if S.shape[1] > time_frames:
            S = S[:, :time_frames]
        elif S.shape[1] < time_frames:
            pad_t = time_frames - S.shape[1]
            S = np.pad(S, ((0, 0), (0, pad_t)), mode="edge")

        S_norm = (S - S.min()) / (S.max() - S.min() + 1e-8)
        return (S_norm * 255).astype(np.uint8)

    def get_generator(self) -> Generator[np.ndarray, None, None]:
        idx = 0
        total = len(self.files)

        while True:
            file_path = self.files[idx]
            try:
                tgt_h, tgt_w, _ = self.config["input_shape"]
                spec = self._audio_to_spectrogram(
                    file_path,
                    self.config["pixel_per_second"],
                    tgt_h,
                    self.config.get("segment_length"),
                )

                spect = np.expand_dims(spec, -1)  # (H, W, 1)
                height, width, _ = spect.shape

                if height != tgt_h:
                    raise ValueError(f"Height mismatch {height} vs {tgt_h}")

                for i in range(0, width // tgt_w):
                    s = i * tgt_w
                    segment = spect[:, s: s + tgt_w]
                    if not np.all(segment == 0):
                        yield segment

            except Exception as e:
                print("SpectrogramGenerator exception:", e, file_path)

            finally:
                idx += 1
                if idx >= total:
                    if self.run_only_once:
                        break
                    idx = 0
                    if self.shuffle:
                        random.shuffle(self.files)

    def __iter__(self):
        return self.get_generator()

    def get_num_files(self) -> int:
        return len(self.files)
