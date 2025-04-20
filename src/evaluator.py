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
Evaluator Class

This module defines the Evaluator class, which provides tools for analyzing
and comparing the tonal characteristics of audio sequences using chroma-based
methods. It includes functionality for calculating pitch class entropy,
comparing average chroma vectors using cosine similarity, and visualizing
chromagrams with matplotlib. These tools support both quantitative and
qualitative evaluation of generated music in a symbolic-to-audio pipeline.
----------------------------------------
"""

import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine

class Evaluator:
    def __init__(self, sr=22050):
        '''
        Initializes the Evaluator with a target sampling rate for audio analysis.

        :param sr: Sampling rate used for audio processing (default is 22050 Hz).
        '''
        self.sr = sr

    def pitch_class_entropy(self, chroma):
        '''
        Calculates the pitch class entropy from a chromagram. This metric reflects
        how evenly distributed pitch classes are, acting as a proxy for tonal diversity.

        :param chroma: A 12-row chromagram array (NumPy array) representing pitch class energy over time.
        :return: Scalar entropy value in bits.
        '''
        p = np.mean(chroma, axis=1)
        p /= np.sum(p)
        return -np.sum(p * np.log2(p + 1e-8))

    def average_chroma_vector(self, chroma):
        '''
        Computes the average chroma vector across time, providing a tonal fingerprint
        of the entire audio segment.

        :param chroma: Chromagram (12 x time) NumPy array.
        :return: 1D NumPy array of length 12 representing average pitch class energy.
        '''
        return np.mean(chroma, axis=1)

    def compare_chroma_similarity(self, chroma1, chroma2):
        '''
        Computes cosine similarity between the average chroma vectors of two
        chromagrams. Used to quantify tonal similarity.

        :param chroma1: First chromagram array (12 x time).
        :param chroma2: Second chromagram array (12 x time).
        :return: Cosine similarity value between 0 and 1 (1 = identical).
        '''
        vec1 = self.average_chroma_vector(chroma1)
        vec2 = self.average_chroma_vector(chroma2)
        return 1 - cosine(vec1, vec2)  # 1 = perfect match, 0 = orthogonal

    def plot_chromagram(self, y, title="Chromagram"):
        '''
        Plots the chromagram of a given audio signal using Librosa and Matplotlib.

        :param y: NumPy array of the audio time series.
        :param title: Title for the plot (default is "Chromagram").
        :return: None
        '''
        chroma = librosa.feature.chroma_stft(y=y, sr=self.sr)
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(chroma, x_axis='time', y_axis='chroma', cmap='coolwarm', sr=self.sr)
        plt.colorbar()
        plt.title(title)
        plt.tight_layout()
        plt.show()
