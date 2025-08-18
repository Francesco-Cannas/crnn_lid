import numpy as np
import scipy.io.wavfile as wav
from crnn_lid.manage.data_loaders.csv_loader import CSVLoader
from numpy.lib import stride_tricks


class Spectrogram2Loader(CSVLoader):

    @staticmethod
    def stft(sig, frameSize, overlapFac=0.5, window=np.hanning):
        win = window(int(frameSize))
        hopSize = int(frameSize - np.floor(overlapFac * frameSize))

        samples = np.append(np.zeros(int(np.floor(frameSize / 2.0))), sig)
        cols = int(np.ceil((len(samples) - frameSize) / float(hopSize))) + 1
        samples = np.append(samples, np.zeros(int(frameSize)))

        frames = stride_tricks.as_strided(
            samples,
            shape=(cols, int(frameSize)),
            strides=(samples.strides[0] * hopSize, samples.strides[0])
        ).copy()
        frames *= win

        return np.fft.rfft(frames)

    @staticmethod
    def logscale_spec(spec, sr=44100, factor=20., alpha=1.0, f0=0.9, fmax=1.0):
        spec = spec[:, 0:128]
        time_bins, freq_bins = np.shape(spec)

        lin = np.linspace(0.0, 1.0, freq_bins)
        scale = np.where(
            lin <= f0,
            lin * alpha,
            ((fmax - alpha * f0) / (fmax - f0)) * (lin - f0) + alpha * f0
        )

        scale *= (freq_bins - 1) / max(scale.max(), 1e-12)

        newspec = np.zeros((time_bins, freq_bins), dtype=np.complex128)

        all_freqs = np.abs(np.fft.fftfreq(freq_bins * 2, 1.0 / sr)[:freq_bins + 1])
        freqs = [0.0] * freq_bins
        totw = [0.0] * freq_bins

        for i in range(freq_bins):
            if i < 1 or i + 1 >= freq_bins:
                newspec[:, i] += spec[:, i]
                freqs[i] += all_freqs[i]
                totw[i] += 1.0
                continue

            j = int(np.floor(scale[i]))

            if j >= freq_bins - 1:
                j = freq_bins - 2
            w_up = float(scale[i] - j)
            w_down = 1.0 - w_up

            newspec[:, j] += w_down * spec[:, i]
            freqs[j] += w_down * all_freqs[i]
            totw[j] += w_down

            newspec[:, j + 1] += w_up * spec[:, i]
            freqs[j + 1] += w_up * all_freqs[i]
            totw[j + 1] += w_up

        for i in range(freq_bins):
            if totw[i] > 1e-6:
                freqs[i] /= totw[i]

        return newspec, freqs

    def create_spectrogram(self, file, bin_size=1024, alpha=1.0):
        sample_rate, samples = wav.read(file)

        if np.issubdtype(samples.dtype, np.integer):
            samples = samples.astype(np.float32) / np.iinfo(samples.dtype).max

        s = self.stft(samples, int(bin_size))
        sshow, _ = self.logscale_spec(s, factor=1, sr=sample_rate, alpha=alpha)
        mag = np.abs(sshow)
        mag = np.maximum(mag, 1e-10)
        ims = 20.0 * np.log10(mag / 1e-5)
        ims = ims.T
        ims = ims[0:128, :]
        return np.expand_dims(ims, -1)

    def process_file(self, file_path):
        spectrogram = self.create_spectrogram(file_path)
        assert len(spectrogram.shape) == 3
        return np.divide(spectrogram, 255.0, dtype=np.float32)
