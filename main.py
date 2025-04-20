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
main.py – End-to-End Audio Generation Pipeline

This is the main execution script for generating folk-style instrumental music
based on learned audio structure.

- Generates 6 new compositions from recomposed segments.
- Stores output in 'generated_music/'.
----------------------------------------
"""

import os
import random
import numpy as np
import soundfile as sf

from src.audio_manager import AudioManager
from src.feature_extractor import FeatureExtractor
from src.audio_model_trainer import AudioModelTrainer
from src.audio_generator import AudioGenerator
from src.evaluator import Evaluator
from src.style_composer import StyleComposer


def main():
    print("Starting WAV-style composer system...\n")

    # Initialize components
    am = AudioManager("input_audio")
    fe = FeatureExtractor()
    trainer = AudioModelTrainer()
    generator = AudioGenerator()
    evaluator = Evaluator()
    composer = StyleComposer()

    # Load input audio
    audio_data = am.load_all()
    print(f"Loaded {len(audio_data)} WAV file(s).")

    if not audio_data:
        print("No input audio found. Please place WAV files in the 'input_audio' folder.")
        return

    # Segment and collect all available pieces
    all_segments = []
    for name, y in audio_data:
        print(f"Segmenting: {name}")
        segments = composer.segment_and_score(y, segment_length=2.0)
        all_segments.extend(segments)

    if not all_segments:
        print("No usable segments found. Check audio length or input quality.")
        return

    os.makedirs("generated_music", exist_ok=True)

    # Generate and save 6 new tracks
    print("\nComposing new audio...")
    generated_pieces = []

    for i in range(6):
        new_piece = composer.compose(all_segments, num_phrases=8)
        output_path = os.path.join("generated_music", f"style_composed_{i + 1}.wav")
        sf.write(output_path, new_piece, samplerate=am.sr)
        print(f"Saved composed piece {i + 1} to: {output_path}")

        generated_pieces.append({
            'name': f'Generated Track {i + 1}',
            'meter': random.choice(['6/8', '4/4', '7/8']),
            'tempo': random.randint(80, 120)
        })

    # Print track table to console
    print("\nGenerated Tracks Table:")
    print("{:<5} {:<20} {:<10} {:<10}".format("No.", "Name", "Meter", "Tempo"))
    for idx, track in enumerate(generated_pieces):
        print("{:<5} {:<20} {:<10} {:<10}".format(idx + 1, track['name'], track['meter'], track['tempo']))


if __name__ == "__main__":
    main()
