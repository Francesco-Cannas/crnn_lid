import librosa
import librosa.feature as lf
import numpy as np

from crnn_lid.manage.data_loaders.csv_loader import CSVLoader


class RosaLoader(CSVLoader):
    SR = 12_000
    N_FFT = 512
    N_MELS = 96
    HOP_LEN = 256

    @staticmethod
    def process_file(file_path: str) -> np.ndarray:
        y, _ = librosa.load(file_path, sr=RosaLoader.SR)

        mel_spec = lf.melspectrogram(
            y=y,
            sr=RosaLoader.SR,
            n_fft=RosaLoader.N_FFT,
            hop_length=RosaLoader.HOP_LEN,
            n_mels=RosaLoader.N_MELS,
            power=2.0
        )

        mel_spec_db = librosa.power_to_db(mel_spec, ref=1.0)

        mel_spec_db = mel_spec_db[..., np.newaxis].astype(np.float32)

        return mel_spec_db
