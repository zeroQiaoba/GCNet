from re import M
import numpy as np  # linear algebra
from tqdm import tqdm
import PIL
import os
import librosa
import random


class default_config:
    sampling_rate = 16000
    duration = 2  # sec
    hop_length = 125 * duration  # to make time steps 128
    fmin = 20
    fmax = sampling_rate // 2
    n_mels = 128
    n_fft = n_mels * 20
    padmode = 'constant'
    samples = sampling_rate * duration


class MelSpecExtractor(object):
    def __init__(self, sampling_rate=None, duration=None, hop_length=None, \
            fmin=None, fmax=None, n_mels=None, n_fft=None, padmode=None, max_samples=None):
        self.sampling_rate = sampling_rate or default_config.sampling_rate
        self.duration = duration or default_config.duration
        self.hop_length = hop_length or default_config.hop_length
        self.fmin = fmin or default_config.fmin
        self.fmax = fmax or default_config.fmax
        self.n_mels = n_mels or default_config.n_mels
        self.n_fft = n_fft or default_config.n_fft
        self.padmode = padmode or default_config.padmode
        self.max_samples = max_samples or default_config.samples
        assert self.max_samples > 0, 'max_samples parameters must be larger than zero'
    

    def read_audio(self, pathname, trim_long_data):
        y, _ = librosa.load(pathname, sr=self.sampling_rate)
        # trim silence
        if 0 < len(y):  # workaround: 0 length causes error
            y, _ = librosa.effects.trim(y)  # trim, top_db=default(60)
        else:
            print(f"found zero length audio {pathname}")
            y = np.zeros((self.max_samples,), np.float32)
        # make it unified length to self.samples
        if len(y) > self.max_samples:  # long enough
            if trim_long_data:
                y = y[0 : self.max_samples]
        else:  # pad blank
            leny = len(y)
            padding = self.max_samples - len(y)  # add padding at both ends
            offset = padding // 2
            y = np.pad(y, (offset, self.max_samples - len(y) - offset), self.padmode)
        return y


    def audio_to_melspectrogram(self, audio):
        spectrogram = librosa.feature.melspectrogram(audio,
                                                    sr=self.sampling_rate,
                                                    n_mels=self.n_mels,
                                                    hop_length=self.hop_length,
                                                    n_fft=self.n_fft,
                                                    fmin=self.fmin,
                                                    fmax=self.fmax)
        spectrogram = librosa.power_to_db(spectrogram)
        spectrogram = spectrogram.astype(np.float32)
        return spectrogram


    def read_as_melspectrogram(self, pathname, trim_long_data=False):
        x = self.read_audio(pathname, trim_long_data)
        mels = self.audio_to_melspectrogram(x)
        return mels
    

    def mono_to_color(self, X, mean=None, std=None, norm_max=None, norm_min=None, eps=1e-6):
        # Stack X as [X,X,X]
        X = np.stack([X, X, X], axis=-1)

        # Standardize
        mean = mean or X.mean()
        X = X - mean
        std = std or X.std()
        Xstd = X / (std + eps)
        _min, _max = Xstd.min(), Xstd.max()
        norm_max = norm_max or _max
        norm_min = norm_min or _min
        if (_max - _min) > eps:
            # Normalize to [0, 255]
            V = Xstd
            V[V < norm_min] = norm_min
            V[V > norm_max] = norm_max
            V = 255 * (V - norm_min) / (norm_max - norm_min)
            V = V.astype(np.uint8)
        else:
            # Just zero
            V = np.zeros_like(Xstd, dtype=np.uint8)
        return V


    def extract(self, wav_path):
        x = self.read_as_melspectrogram(wav_path, trim_long_data=False)
        x_color = self.mono_to_color(x)
        return x_color


if __name__ == '__main__':
    extractor = MelSpecExtractor()
    # wav_path = '/data3/lrc/IEMOCAP_full_release/Session1/sentences/wav/Ses01F_script03_2/Ses01F_script03_2_M001.wav'
    wav_path = '/data3/lrc/IEMOCAP_full_release/Session1/sentences/wav/Ses01F_script03_2/Ses01F_script03_2_M026.wav'
    melspec = extractor.extract(wav_path)
    print(melspec.shape)