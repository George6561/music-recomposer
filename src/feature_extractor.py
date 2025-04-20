"""
MIT License

Copyright (c) 2025 George Miller

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

----------------------------------------
FeatureExtractor Class

This module defines the FeatureExtractor class, which provides functionality
for extracting a variety of audio features from waveform input using Librosa.
Extracted features include MFCCs, chroma vectors, spectral centroid, contrast,
zero-crossing rate, and short-time Fourier transform magnitude. These are used
as inputs for training, evaluation, or classification in generative music or
audio modeling systems.
----------------------------------------
"""


import librosa
import numpy as np

class FeatureExtractor:
    def __init__(self, sr=22050):
        '''
        Initializes the FeatureExtractor with a target sampling rate for analysis.

        :param sr: Sampling rate for feature extraction (default is 22050 Hz).
        '''
        self.sr = sr

    def extract_features(self, y):
        '''
        Extracts a comprehensive set of audio features from a time-domain audio signal.
        Returns a dictionary containing the following:
            - STFT magnitude
            - MFCCs
            - Chroma (pitch class energy)
            - Spectral centroid (brightness)
            - Spectral contrast (harmonicity vs noise)
            - Zero crossing rate (percussiveness)

        :param y: 1D NumPy array representing the audio time series.
        :return: Dictionary of extracted audio features.
        '''
        features = {}

        # Basic time-frequency analysis
        stft = np.abs(librosa.stft(y))
        features["stft"] = stft

        # MFCCs (Timbre, low-level texture)
        features["mfcc"] = librosa.feature.mfcc(y=y, sr=self.sr, n_mfcc=13)

        # Chroma (Pitch Class energy)
        features["chroma"] = librosa.feature.chroma_stft(S=stft, sr=self.sr)

        # Spectral Centroid (brightness)
        features["centroid"] = librosa.feature.spectral_centroid(S=stft, sr=self.sr)

        # Spectral Contrast (harmonicity vs noise)
        features["contrast"] = librosa.feature.spectral_contrast(S=stft, sr=self.sr)

        # Zero Crossing Rate (percussiveness)
        features["zcr"] = librosa.feature.zero_crossing_rate(y)

        return features
