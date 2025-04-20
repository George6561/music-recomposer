# Music Recomposition System 🎶

Author: George Miller (gnhp49)  
Course: COMP3721 – Machine Learning for Music  
Project: Symbolic recomposition of WAV audio using audio segmentation and scoring

---

## 🧠 Overview

This project generates new music by segmenting and recombining real audio files (WAV format). It uses audio-based structure learning to produce stylistically coherent, original pieces.

The system performs:
- Audio segmentation and scoring
- Phrase recomposition
- WAV output generation
- Optional visual analysis (chroma, spectrogram, pitch histograms)

---

## 📂 Folder Structure

- `src/` – all source code modules
- `main.py` – entry point script
- `input_audio/` – input WAV files (a small subset; full dataset linked below)
- `generated_music/` – output compositions
- `analysis_plots/` – optional visualizations
- `models/` – any saved models or data
- `doc/` – final report, if included
- `video/` – optional demo video

---

## ▶️ How to Run

1. Place your `.wav` files in `input_audio/`
2. Run `main.py`  
3. Outputs will be saved to `generated_music/`

Optionally, visualizations will be generated in `analysis_plots/`.

---

## 🔗 Full Input Dataset

To keep the repo size manageable, only a few WAV files are included.  
The full input set (20 tracks) is available here:

📁 [Google Drive Folder](https://drive.google.com/drive/folders/1Ew8-c5CaMLMBSjjA0gDtfwNlJT4lhwUp)

---

## 📝 Citation

This project was developed as part of COMP3721: Machine Learning for Music.  
All code is released under the MIT License.

