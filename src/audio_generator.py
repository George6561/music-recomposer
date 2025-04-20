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
AudioGenerator Class

This module defines the AudioGenerator class, which performs simple, randomized
audio transformations on a folder of WAV files to simulate musical generation.
Transformations include random pitch shifting and time stretching using Librosa.
Generated audio is saved to a designated output directory in WAV format.
Intended for use in symbolic-to-audio postprocessing or augmentation.
----------------------------------------
"""



import os
import librosa
import soundfile as sf
import numpy as np
import random


class AudioGenerator:

    def __init__(self, input_dir="input_audio", output_dir="generated_music", sr=22050):
        '''
        Initializes the AudioGenerator with specified input and output directories and sampling rate.

        :param input_dir: Directory containing input WAV files to be transformed.
        :param output_dir: Directory where generated audio files will be saved.
        :param sr: Sampling rate for audio processing (default is 22050 Hz).
        '''
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.sr = sr
        os.makedirs(self.output_dir, exist_ok=True)

    def _transform_audio(self, y):
        '''
        Applies a simple transformation to the input audio by randomly altering pitch and tempo.

        :param y: NumPy array containing the audio time series.
        :return: Transformed audio signal with pitch shift and time stretch applied.
        '''
        pitch_shift = random.choice([-2, -1, 1, 2])
        time_stretch = random.uniform(0.9, 1.1)

        y_shifted = librosa.effects.pitch_shift(y, sr=self.sr, n_steps=pitch_shift)
        y_stretched = librosa.effects.time_stretch(y_shifted, rate=time_stretch)

        return y_stretched

    def generate(self):
        '''
        Processes all WAV files in the input directory, applies transformations, and saves
        the resulting audio to the output directory with modified filenames.

        :return: None
        '''
        print("Generating audio based on existing inputs...")
        for file in os.listdir(self.input_dir):
            if file.lower().endswith(".wav"):
                input_path = os.path.join(self.input_dir, file)
                y, _ = librosa.load(input_path, sr=self.sr)

                # Apply transformation
                y_gen = self._transform_audio(y)

                # Save to output
                out_name = f"generated_{file}"
                output_path = os.path.join(self.output_dir, out_name)
                sf.write(output_path, y_gen, self.sr)
                print(f"Generated: {output_path}")
