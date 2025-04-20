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
AudioModelTrainer Class

This module defines the AudioModelTrainer class, a mock framework for training
on extracted audio features such as MFCCs. It simulates a learning process by
calculating the average feature vector across a dataset and saves the result
as a placeholder model using Python's pickle module. While simplistic, this
structure is extensible and can be adapted to train real models such as VAEs,
transformers, or DDSP-based systems in a music generation pipeline.
----------------------------------------
"""

import numpy as np
import os
import pickle

class AudioModelTrainer:
    def __init__(self, output_dir="models"):
        '''
        Initializes the AudioModelTrainer with a target directory to save or load models.

        :param output_dir: Directory where trained models will be saved (default is "models").
        '''
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.model = None  # Placeholder for a real model

    def train(self, dataset):
        '''
        Simulates training on a dataset by computing the average MFCC feature vector.
        This is a placeholder for more advanced training procedures like VAE or transformer-based models.

        :param dataset: A list of feature dictionaries, each containing at least an "mfcc" key.
        :return: None
        '''
        print("Starting mock training...")
        all_mfccs = [features["mfcc"] for features in dataset]
        mfcc_stack = np.hstack(all_mfccs)

        # Simulate "learning" by calculating mean feature vector
        mean_vector = np.mean(mfcc_stack, axis=1)
        self.model = {"mean_mfcc": mean_vector}

        # Save the model
        self.save_model("mock_model.pkl")
        print("Model training complete and saved.")

    def save_model(self, filename="mock_model.pkl"):
        '''
        Saves the current model (dictionary object) to a file using Python's pickle module.

        :param filename: Name of the file to save the model to (default is "mock_model.pkl").
        :return: None
        '''
        path = os.path.join(self.output_dir, filename)
        with open(path, "wb") as f:
            pickle.dump(self.model, f)
        print(f"Saved model to {path}")

    def load_model(self, filename="mock_model.pkl"):
        '''
        Loads a previously saved model from the specified file.

        :param filename: Name of the model file to load (default is "mock_model.pkl").
        :return: None
        :raises FileNotFoundError: If the specified model file does not exist.
        '''
        path = os.path.join(self.output_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model not found at {path}")
        with open(path, "rb") as f:
            self.model = pickle.load(f)
        print(f"Loaded model from {path}")

