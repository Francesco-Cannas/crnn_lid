import logging
import random
import threading
from pathlib import Path
from queue import Queue, Empty
from typing import Generator, List, Optional

import librosa
import librosa.feature as lf
import librosa.util.exceptions as lue
import numpy as np

logging.basicConfig(
    filename='spectrogram_generator.log',
    level=logging.ERROR,
    format='%(asctime)s %(levelname)s:%(name)s:%(message)s'
)
logger = logging.getLogger(__name__)

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
            max_queue_size: int = 100,
            run_only_once: bool = False,
    ) -> None:
        self.source = Path(source)
        self.config = config
        self.shuffle = shuffle
        self.run_only_once = run_only_once

        self.overlap_ratio: float = float(config.get('overlap_ratio', 0.0))

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

        self.queue: "Queue[np.ndarray]" = Queue(max_queue_size)
        self._stop_event = threading.Event()

        self._worker_thread = threading.Thread(
            target=self._fill_queue,
            daemon=True
        )
        self._worker_thread.start()

    def _fill_queue(self) -> None:

        idx = 0
        total = len(self.files)
        while not self._stop_event.is_set():
            file_path = self.files[idx]
            try:
                tgt_h, tgt_w, _ = self.config["input_shape"]
                spec = self._audio_to_spectrogram(
                    file_path,
                    self.config["pixel_per_second"],
                    tgt_h,
                    self.config.get("segment_length"),
                )
                spect = np.expand_dims(spec, -1)
                height, width, _ = spect.shape

                if height != tgt_h:
                    raise ValueError(f"Height mismatch {height} vs {tgt_h}")

                step = max(1, int(tgt_w * (1 - self.overlap_ratio)))
                for start in range(0, width - tgt_w + 1, step):
                    segment = spect[:, start: start + tgt_w]
                    if not np.all(segment == 0):
                        self.queue.put(segment)

            except lue.ParameterError as e:
                logger.error("Librosa ParameterError on %s: %s", file_path, e)
            except ValueError as e:
                logger.error("ValueError on %s: %s", file_path, e)
            except Exception as e:
                logger.exception("Unexpected error processing %s", file_path, e)

            idx += 1
            if idx >= total:
                if self.run_only_once:
                    break
                idx = 0
                if self.shuffle:
                    random.shuffle(self.files)

        self._stop_event.set()

    def get_generator(self, timeout: Optional[float] = None) -> Generator[np.ndarray, None, None]:
        while not (self.run_only_once and self._stop_event.is_set() and self.queue.empty()):
            try:
                seg = self.queue.get(timeout=timeout)
                yield seg
            except Empty:
                if self.run_only_once and self._stop_event.is_set():
                    break
                continue

    def __iter__(self):
        return self.get_generator()

    def get_num_files(self) -> int:
        return len(self.files)

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
