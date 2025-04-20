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
Visualizer Class

This module defines the Visualizer class, which creates visual representations
of audio analysis features. It includes plotting utilities for chromagrams,
spectrograms, and pitch class histograms using Librosa and Matplotlib.
Plots are saved as image files for use in analysis reports, presentations, or
system evaluation. Designed to support both qualitative review and publication
of symbolic-to-audio results.
----------------------------------------
"""

import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import os

class Visualizer:
    def __init__(self, sr=22050, output_dir="analysis_plots"):
        '''
        Initializes the Visualizer with a sampling rate and output directory for saving plots.

        :param sr: Sampling rate for audio processing (default is 22050 Hz).
        :param output_dir: Directory where visualizations will be saved (default is "analysis_plots").
        '''
        self.sr = sr
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir

    def save_plot(self, fig, name):
        '''
        Saves a Matplotlib figure to the output directory and closes it.

        :param fig: Matplotlib figure object to save.
        :param name: Filename for the saved image (including extension, e.g., "plot.png").
        :return: None
        '''
        path = os.path.join(self.output_dir, name)
        fig.savefig(path, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved plot: {path}")

    def plot_chromagram(self, y, title="Chromagram", filename="chroma.png"):
        '''
        Generates and saves a chromagram plot from an audio signal.

        :param y: Audio time series as a NumPy array.
        :param title: Title of the plot (default is "Chromagram").
        :param filename: Filename to save the image as (default is "chroma.png").
        :return: None
        '''
        chroma = librosa.feature.chroma_stft(y=y, sr=self.sr)
        fig, ax = plt.subplots(figsize=(10, 4))
        img = librosa.display.specshow(chroma, x_axis='time', y_axis='chroma', cmap='coolwarm', ax=ax)
        fig.colorbar(img, ax=ax)
        ax.set_title(title)
        self.save_plot(fig, filename)

    def plot_spectrogram(self, y, title="Spectrogram", filename="spec.png"):
        '''
        Generates and saves a spectrogram plot in decibel scale from an audio signal.

        :param y: Audio time series as a NumPy array.
        :param title: Title of the plot (default is "Spectrogram").
        :param filename: Filename to save the image as (default is "spec.png").
        :return: None
        '''
        S = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        fig, ax = plt.subplots(figsize=(10, 4))
        img = librosa.display.specshow(S, x_axis='time', y_axis='log', sr=self.sr, ax=ax)
        fig.colorbar(img, ax=ax, format='%+2.0f dB')
        ax.set_title(title)
        self.save_plot(fig, filename)

    def plot_pitch_class_histogram(self, chroma, title="Pitch Class Histogram", filename="pchist.png"):
        '''
        Generates and saves a bar plot of the average pitch class energy distribution.

        :param chroma: Chromagram (12 x time) NumPy array.
        :param title: Title of the plot (default is "Pitch Class Histogram").
        :param filename: Filename to save the image as (default is "pchist.png").
        :return: None
        '''
        pitch_class_profile = np.mean(chroma, axis=1)
        fig, ax = plt.subplots()
        ax.bar(np.arange(12), pitch_class_profile)
        ax.set_xticks(np.arange(12))
        ax.set_xticklabels(['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'])
        ax.set_title(title)
        ax.set_ylabel("Normalized Energy")
        self.save_plot(fig, filename)
