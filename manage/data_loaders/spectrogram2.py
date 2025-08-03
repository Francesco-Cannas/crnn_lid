import numpy as np
import scipy.io.wavfile as wav
from numpy.lib import stride_tricks

from crnn_lid.manage.data_loaders.csv_loader import CSVLoader


class Spectrogram2Loader(CSVLoader):

    @staticmethod
    def stft(sig, frameSize, overlapFac=0.5, window=np.hanning):
        win = window(frameSize)
        hopSize = int(frameSize - np.floor(overlapFac * frameSize))

        samples = np.append(np.zeros(np.floor(frameSize / 2.0)), sig)
        cols = np.ceil((len(samples) - frameSize) / float(hopSize)) + 1
        samples = np.append(samples, np.zeros(frameSize))

        frames = stride_tricks.as_strided(samples, shape=(cols, frameSize),
                                          strides=(samples.strides[0] * hopSize, samples.strides[0])).copy()
        frames *= win

        return np.fft.rfft(frames)

    @staticmethod
    def logscale_spec(spec, sr=44100, factor=20., alpha=1.0, f0=0.9, fmax=1):
        spec = spec[:, 0:128]
        time_bins, freq_bins = np.shape(spec)
        scale = np.linspace(0, 1, freq_bins)

        scale = np.array(
            map(lambda x: x * alpha if x <= f0 else (fmax - alpha * f0) / (fmax - f0) * (x - f0) + alpha * f0, scale))
        scale *= (freq_bins - 1) / max(scale)

        newspec = np.complex128(np.zeros([time_bins, freq_bins]))
        all_freqs = np.abs(np.fft.fftfreq(freq_bins * 2, 1. / sr)[:freq_bins + 1])
        freqs = [0.0] * freq_bins
        totw = [0.0] * freq_bins

        for i in range(0, freq_bins):
            if i < 1 or i + 1 >= freq_bins:
                newspec[:, i] += spec[:, i]
                freqs[i] += all_freqs[i]
                totw[i] += 1.0
                continue
            else:
                w_up = scale[i] - np.floor(scale[i])
                w_down = 1 - w_up
                j = int(np.floor(scale[i]))

                newspec[:, j] += w_down * spec[:, i]
                freqs[j] += w_down * all_freqs[i]
                totw[j] += w_down

                newspec[:, j + 1] += w_up * spec[:, i]
                freqs[j + 1] += w_up * all_freqs[i]
                totw[j + 1] += w_up

        for i in range(len(freqs)):
            if totw[i] > 1e-6:
                freqs[i] /= totw[i]

        return newspec, freqs

    def create_spectrogram(self, file, bin_size=1024, alpha=1):

        sample_rate, samples = wav.read(file)
        s = self.stft(samples, bin_size)

        sshow, freq = self.logscale_spec(s, factor=1, sr=sample_rate, alpha=alpha)
        sshow = sshow[2:, :]
        ims = 20. * np.log10(np.abs(sshow) / 10e-6)

        ims = np.transpose(ims)
        ims = ims[0:128, :]

        return np.expand_dims(ims, -1)

    def process_file(self, file_path):

        spectrogram = self.create_spectrogram(file_path)

        assert len(spectrogram.shape) == 3

        return np.divide(spectrogram, 255.0)
