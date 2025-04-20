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
StyleComposer Class

This module defines the StyleComposer class, which provides simple but flexible
tools for musical audio generation through recombination. It segments input
audio into equal-length phrases, scores or filters them, and then applies
stochastic variation (e.g., pitch shifting, time stretching) during recomposition.
Useful for prototyping algorithmic composition or data augmentation workflows
that blend existing stylistic material into new musical outputs.
----------------------------------------
"""

import numpy as np
import random
import librosa

class StyleComposer:
    def __init__(self, sr=22050):
        '''
        Initializes the StyleComposer with a given audio sampling rate.

        :param sr: Sampling rate for all audio operations (default is 22050 Hz).
        '''
        self.sr = sr

    def segment_and_score(self, y, segment_length=2.0):
        '''
        Splits an input audio signal into fixed-length segments.

        :param y: NumPy array containing the input audio time series.
        :param segment_length: Duration (in seconds) of each segment (default is 2.0 seconds).
        :return: List of equally sized audio segments as NumPy arrays.
        '''
        samples_per_seg = int(segment_length * self.sr)
        segments = [y[i:i+samples_per_seg] for i in range(0, len(y), samples_per_seg)]
        return [seg for seg in segments if len(seg) == samples_per_seg]

    def compose(self, segments, variation_strength=0.1, num_phrases=6):
        '''
        Generates a new audio sequence by randomly selecting and applying variation
        to segments, creating a recomposed audio output.

        :param segments: List of audio segments to sample from.
        :param variation_strength: Not currently used, reserved for future parameter tuning (default is 0.1).
        :param num_phrases: Number of segments to include in the generated audio (default is 6).
        :return: A single NumPy array representing the concatenated, recomposed audio.
        '''
        new_audio = []

        for _ in range(num_phrases):
            seg = random.choice(segments)

            # Apply slight pitch shift or time stretch
            if random.random() < 0.5:
                seg = librosa.effects.pitch_shift(seg, sr=self.sr, n_steps=random.uniform(-1, 1))
            else:
                seg = librosa.effects.time_stretch(seg, rate=random.uniform(0.95, 1.05))

            new_audio.append(seg)

        return np.concatenate(new_audio)
