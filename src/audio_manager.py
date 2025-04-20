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
AudioManager Class

This module defines the AudioManager class, responsible for handling audio
file input/output. It supports loading all or individual WAV files from a
specified directory, trimming silence using Librosa, and saving audio arrays
to disk. Intended for preprocessing audio input or managing intermediate
results in a generative audio pipeline.
----------------------------------------
"""


import os
import librosa
import soundfile as sf

class AudioManager:
    def __init__(self, input_dir="input_audio", sr=22050):
        '''
        Initializes the AudioManager with a directory of input audio files and a sampling rate.

        :param input_dir: Directory containing input WAV files to be loaded.
        :param sr: Sampling rate for loading audio (default is 22050 Hz).
        '''
        self.input_dir = input_dir
        self.sr = sr

    def load_all(self):
        '''
        Loads and trims silence from all WAV files in the input directory.

        :return: A list of tuples, each containing (filename, trimmed audio array).
        '''
        audio_data = []
        for file in os.listdir(self.input_dir):
            if file.lower().endswith(".wav"):
                file_path = os.path.join(self.input_dir, file)
                y, _ = librosa.load(file_path, sr=self.sr, mono=True)
                y_trimmed, _ = librosa.effects.trim(y)
                audio_data.append((file, y_trimmed))
        return audio_data

    def load_file(self, file_name):
        '''
        Loads and trims silence from a specific WAV file by name.

        :param file_name: Name of the WAV file to load from the input directory.
        :return: Trimmed audio time series as a NumPy array.
        :raises FileNotFoundError: If the specified file does not exist in the input directory.
        '''
        path = os.path.join(self.input_dir, file_name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"File {file_name} not found in {self.input_dir}")
        y, _ = librosa.load(path, sr=self.sr, mono=True)
        return librosa.effects.trim(y)[0]

    def save_audio(self, y, file_name, output_dir="generated_music"):
        '''
        Saves an audio signal to a WAV file in the specified output directory.

        :param y: NumPy array containing the audio signal to save.
        :param file_name: Name of the output WAV file.
        :param output_dir: Directory to save the file (created if it doesn't exist).
        :return: Full path to the saved output file.
        '''
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, file_name)
        sf.write(output_path, y, self.sr)
        print(f"Saved audio: {output_path}")
        return output_path
